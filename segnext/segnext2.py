import json
from abc import abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


"""
    逐层卷积
"""


class DepthwiseConv(nn.Module):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小，元组类型
        padding: 补充
        stride: 步长
    """

    def __init__(self, in_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1), bias=False):
        super(DepthwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            groups=in_channels,
            bias=bias
        )

    def forward(self, x):
        out = self.conv(x)
        return out


"""
    逐点卷积
"""


class PointwiseConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(PointwiseConv, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


"""
    深度可分离卷积
"""


class DepthwiseSeparableConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), padding=(1, 1), stride=(1, 1)):
        super(DepthwiseSeparableConv, self).__init__()

        self.conv1 = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

        self.conv2 = PointwiseConv(
            in_channels=in_channels,
            out_channels=out_channels
        )

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out


"""
    下采样
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // stride, width // stride]
"""


class DownSampling(nn.Module):
    """
        in_channels: 输入通道数
        out_channels: 输出通道数
        kernel_size: 卷积核大小
        stride: 步长
        norm_layer: 正则化层，如果为None，使用BatchNorm
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm_layer=None):
        super(DownSampling, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size[0] // 2, kernel_size[-1] // 2)
        )

        if norm_layer is None:
            self.norm = nn.BatchNorm2d(num_features=out_channels)
        else:
            self.norm = norm_layer

    def forward(self, x):
        out = self.conv(x)
        out = self.norm(out)
        return out


class _MatrixDecomposition2DBase(nn.Module):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 100,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(_MatrixDecomposition2DBase, self).__init__()
        args: dict = json.loads(args)
        for k, v in args.items():
            setattr(self, k, v)

    @abstractmethod
    def _build_bases(self, batch_size):
        pass

    @abstractmethod
    def local_step(self, x, bases, coef):
        pass

    @torch.no_grad()
    def local_inference(self, x, bases):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batchszie * MD_S, N, MD_R)
        coef = torch.bmm(x.transpose(1, 2), bases)
        coef = F.softmax(self.INV_T * coef, dim=-1)

        steps = self.TRAIN_STEPS if self.training else self.EVAL_STEPS
        for _ in range(steps):
            bases, coef = self.local_step(x, bases, coef)

        return bases, coef

    @abstractmethod
    def compute_coef(self, x, bases, coef):
        pass

    def forward(self, x):

        batch_size, channels, height, width = x.shape

        # (batch_size, channels, height, width) -> (batch_size * MD_S, MD_D, N)
        if self.SPATIAL:
            self.MD_D = channels // self.MD_S
            N = height * width
            x = x.view(batch_size * self.MD_S, self.MD_D, N)
        else:
            self.MD_D = height * width
            N = channels // self.MD_S
            x = x.view(batch_size * self.MD_S, N, self.MD_D).transpose(1, 2)

        if not self.RAND_INIT and not hasattr(self, 'bases'):
            bases = self._build_bases(1)
            self.register_buffer('bases', bases)

        # (MD_S, MD_D, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        if self.RAND_INIT:
            bases = self._build_bases(batch_size)
        else:
            bases = self.bases.repeat(batch_size, 1, 1)

        bases, coef = self.local_inference(x, bases)

        # (batch_size * MD_S, N, MD_R)
        coef = self.compute_coef(x, bases, coef)

        # (batch_size * MD_S, MD_D, MD_R) @ (batch_size * MD_S, N, MD_R)^T -> (batch_size * MD_S, MD_D, N)
        x = torch.bmm(bases, coef.transpose(1, 2))

        # (batch_size * MD_S, MD_D, N) -> (batch_size, channels, height, width)
        if self.SPATIAL:
            x = x.view(batch_size, channels, height, width)
        else:
            x = x.transpose(1, 2).view(batch_size, channels, height, width)

        # (batch_size * height, MD_D, MD_R) -> (batch_size, height, N, MD_D)
        bases = bases.view(batch_size, self.MD_S, self.MD_D, self.MD_R)

        if self.return_bases:
            return x, bases
        return x


class NMF2D(_MatrixDecomposition2DBase):
    def __init__(
            self,
            args=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(NMF2D, self).__init__(args)

    def _build_bases(self, batch_size):
        bases = torch.rand((batch_size * self.MD_S, self.MD_D, self.MD_R)).to(self.device)
        bases = F.normalize(bases, dim=1)

        return bases

    # @torch.no_grad()
    def local_step(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ [(batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)]
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # Multiplicative Update
        coef = coef * numerator / (denominator + 1e-6)

        # (batch_size * MD_S, MD_D, N) @ (batch_size * MD_S, N, MD_R) -> (batch_size * MD_S, MD_D, MD_R)
        numerator = torch.bmm(x, coef)
        # (batch_size * MD_S, MD_D, MD_R) @ [(batch_size * MD_S, N, MD_R)^T @ (batch_size * MD_S, N, MD_R)]
        # -> (batch_size * MD_S, D, MD_R)
        denominator = bases.bmm(coef.transpose(1, 2).bmm(coef))
        # Multiplicative Update
        bases = bases * numerator / (denominator + 1e-6)

        return bases, coef

    def compute_coef(self, x, bases, coef):
        # (batch_size * MD_S, MD_D, N)^T @ (batch_size * MD_S, MD_D, MD_R) -> (batch_size * MD_S, N, MD_R)
        numerator = torch.bmm(x.transpose(1, 2), bases)
        # (batch_size * MD_S, N, MD_R) @ (batch_size * MD_S, MD_D, MD_R)^T @ (batch_size * MD_S, MD_D, MD_R)
        # -> (batch_size * MD_S, N, MD_R)
        denominator = coef.bmm(bases.transpose(1, 2).bmm(bases))
        # multiplication update
        coef = coef * numerator / (denominator + 1e-6)
        return coef


import math
from enum import Enum

import numpy as np
import torch
import torch.optim as optim


class SchedulerType(Enum):
    STEP_SCHEDULER = "step",
    MULTI_STEP_SCHEDULER = "multi_step",
    EXPONENTIAL_SCHEDULER = "exponential",
    COSINE_ANNEALING_SCHEDULER = "cosine_annealing",
    LINEAR_WARMUP_THEN_POLY_SCHEDULER = "linear_warmup_then_poly"


class StepScheduler:
    """
        optimizer: 优化器
        step_size: 每间隔多少步，就去计算优化器的学习率并将其更新
        gamma: lr_(t+1) = lr_(t) * gamma
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """

    def __init__(self, optimizer, step_size=30, gamma=0.1, verbose=False):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.StepLR(
            optimizer=self.optimizer,
            step_size=self.step_size,
            gamma=self.gamma,
            last_epoch=-1,
            verbose=self.verbose
        )

    """
        调用学习率调度器
    """

    def step(self):
        self.lr_scheduler.step()

    """
        获得学习率调度器的状态
    """

    def get_state_dict(self):
        return self.lr_scheduler.state_dict()

    """
        加载学习率调度器的状态字典
    """

    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)


class MultiStepScheduler:
    """
        optimizer: 优化器
        milestones: 列表，列表内的数据必须是整数且递增，每一个数表示调度器被执行了对应次数后，就更新优化器的学习率
        gamma: lr_(t+1) = lr_(t) * gamma
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """

    def __init__(self, optimizer, milestones, gamma, verbose=False):
        self.optimizer = optimizer
        self.milestones = milestones
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer=self.optimizer,
            milestones=self.milestones,
            gamma=gamma,
            last_epoch=-1,
            verbose=self.verbose
        )

    """
        调用学习率调度器
    """

    def step(self):
        self.lr_scheduler.step()

    """
        获得学习率调度器的状态
    """

    def get_state_dict(self):
        return self.lr_scheduler.state_dict()

    """
        加载学习率调度器的状态字典
    """

    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)


class ExponentialScheduler:
    """
        optimizer: 优化器
        gamma: lr_(t+1) = lr_(t) * gamma, 每一次调用，优化器的学习率都会更新
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """

    def __init__(self, optimizer, gamma=0.95, verbose=False):
        self.optimizer = optimizer
        self.gamma = gamma
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer,
            gamma=self.gamma,
            last_epoch=-1,
            verbose=self.verbose
        )

        """
            调用学习率调度器
        """

        def step(self):
            self.lr_scheduler.step()

        """
            获得学习率调度器的状态
        """

        def get_state_dict(self):
            return self.lr_scheduler.state_dict()

        """
            加载学习率调度器的状态字典
        """

        def load_state_dict(self, state_dict: dict):
            self.lr_scheduler.load_state_dict(state_dict)


class CosineAnnealingScheduler:
    """
        optimizer: 优化器，优化器中有一个已经设定的初始学习率，这个初始学习率就是调度器能达到的最大学习率(max_lr)
        t_max: 周期，调度器每被调用2 * t_max，优化器的学习率就会从max_lr -> min_lr -> max_lr
        min_lr: 最小学习率
        verbose: 是否跟踪学习率的变化并打印到控制台中，默认False(不跟踪)
    """

    def __init__(self, optimizer, t_max=5, min_lr=0, verbose=False):
        self.optimizer = optimizer
        self.t_max = t_max
        self.min_lr = min_lr
        self.verbose = verbose
        self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=self.optimizer,
            T_max=self.t_max,
            eta_min=self.min_lr,
            last_epoch=-1,
            verbose=self.verbose
        )

    """
        调用学习率调度器
    """

    def step(self):
        self.lr_scheduler.step()

    """
        获得学习率调度器的状态
    """

    def get_state_dict(self):
        return self.lr_scheduler.state_dict()

    """
        加载学习率调度器的状态字典
    """

    def load_state_dict(self, state_dict: dict):
        self.lr_scheduler.load_state_dict(state_dict)


class LinearWarmupThenPolyScheduler:
    """
        预热阶段采用Linear，之后采用Poly
        optimizer: 优化器
        warmup_iters: 预热步数
        total_iters: 总训练步数
        min_lr: 最低学习率
    """

    def __init__(self, optimizer, warmup_iters=1500, total_iters=2000, warmup_ratio=1e-6, min_lr=0., power=1.):
        self.optimizer = optimizer
        self.current_iters = 0
        self.warmup_iters = warmup_iters
        self.total_iters = total_iters
        self.warmup_ration = warmup_ratio
        self.min_lr = min_lr
        self.power = power

        self.base_lr = None
        self.regular_lr = None
        self.warmup_lr = None

    def get_base_lr(self):
        return np.array(
            [param_group.setdefault("initial_lr", param_group["lr"]) for param_group in self.optimizer.param_groups])

    def get_lr(self):
        coeff = (1 - self.current_iters / self.total_iters) ** self.power
        return (self.base_lr - np.full_like(self.base_lr, self.min_lr)) * coeff + np.full_like(self.base_lr,
                                                                                               self.min_lr)

    def get_regular_lr(self):
        return self.get_lr()

    def get_warmup_lr(self):
        k = (1 - self.current_iters / self.warmup_iters) * (1 - self.warmup_ration)
        return (1 - k) * self.regular_lr

    def update(self):
        assert 0 <= self.current_iters < self.total_iters
        self.current_iters = self.current_iters + 1
        self.base_lr = self.get_base_lr()
        self.regular_lr = self.get_regular_lr()
        self.warmup_lr = self.get_warmup_lr()

    def set_lr(self):
        if self.current_iters <= self.warmup_iters:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.warmup_lr[idx]
        elif self.current_iters <= self.total_iters:
            for idx, param_group in enumerate(self.optimizer.param_groups):
                param_group["lr"] = self.regular_lr[idx]

    def step(self):
        self.update()
        self.set_lr()


"""
    获取学习率调度器
    optimizer: 使用学习率调度器的优化器
    scheduler_type: 要获取的调度器的类型
    kwargs: 参数字典，作用于调度器

    需要改变优化器的参数，在该方法中调整
"""


def get_lr_scheduler(optimizer: optim, scheduler_type: SchedulerType, kwargs=None):
    if kwargs is None:
        # 返回默认设置的调度器
        if scheduler_type == SchedulerType.STEP_SCHEDULER:
            return StepScheduler(
                optimizer=optimizer,
                step_size=30,
                gamma=0.1,
                verbose=False
            )
        elif scheduler_type == SchedulerType.MULTI_STEP_SCHEDULER:
            return MultiStepScheduler(
                optimizer=optimizer,
                milestones=[30, 60, 90],
                gamma=0.1,
                verbose=False
            )
        elif scheduler_type == SchedulerType.EXPONENTIAL_SCHEDULER:
            return ExponentialScheduler(
                optimizer=optimizer,
                gamma=0.95,
                verbose=False
            )
        elif scheduler_type == SchedulerType.COSINE_ANNEALING_SCHEDULER:
            return CosineAnnealingScheduler(
                optimizer=optimizer,
                t_max=5,
                min_lr=0,
                verbose=False
            )
        elif scheduler_type == SchedulerType.LINEAR_WARMUP_THEN_POLY_SCHEDULER:
            return LinearWarmupThenPolyScheduler(
                optimizer=optimizer,
                warmup_iters=1500,
                total_iters=2000,
                warmup_ratio=1e-6,
                min_lr=0.,
                power=1.
            )
    else:
        # 返回自定义设置的调度器
        if scheduler_type == SchedulerType.STEP_SCHEDULER:
            return StepScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.MULTI_STEP_SCHEDULER:
            return MultiStepScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.EXPONENTIAL_SCHEDULER:
            return ExponentialScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.COSINE_ANNEALING_SCHEDULER:
            return CosineAnnealingScheduler(
                optimizer=optimizer,
                **kwargs
            )
        elif scheduler_type == SchedulerType.LINEAR_WARMUP_THEN_POLY_SCHEDULER:
            return LinearWarmupThenPolyScheduler(
                optimizer=optimizer,
                **kwargs
            )


import copy
import math
import os.path
from pathlib import Path
import torch.nn as nn
import torch
import yaml
# import model
import json
import re
import torch.optim as optim

# import learning_rate_scheduler

"""
    获取模型
    @:param train: 是否获取模型进行训练
                   如果为True，使用模型进行训练；
                   如果为False，使用模型进行预测。
    @:param model_config: 模型配置文件路径
    @:param train_config: 训练配置文件路径
    @:param predict_config: 预测配置文件路径
    @:return 实例化模型
"""


def get_model(
        train: bool,
        model_config=Path("config") / "model.yaml",
        train_config=Path("config") / "train.yaml",
        predict_config=Path("config") / "predict.yaml"
):
    with model_config.open("r", encoding="utf-8") as mcf:
        model_config = yaml.load(mcf, Loader=yaml.FullLoader)

        nmf2d_config = model_config["nmf2d_config"]
        if train:
            with train_config.open("r", encoding="utf-8") as tcf:
                train_config = yaml.load(tcf, Loader=yaml.FullLoader)
                device = train_config["device"]
        else:
            with predict_config.open("r", encoding="utf-8") as pcf:
                predict_config = yaml.load(pcf, Loader=yaml.FullLoader)
                device = predict_config["device"]
        nmf2d_config["device"] = device

        net = model.SegNeXt(
            embed_dims=model_config["embed_dims"],
            expand_rations=model_config["expand_rations"],
            depths=model_config["depths"],
            drop_prob_of_encoder=model_config["drop_prob_of_encoder"],
            drop_path_prob=model_config["drop_path_prob"],
            hidden_channels=model_config["channels_of_hamburger"],
            out_channels=model_config["channels_of_hamburger"],
            classes_num=len(model_config["classes"]),
            drop_prob_of_decoder=model_config["drop_prob_of_decoder"],
            nmf2d_config=json.dumps(nmf2d_config)
        ).to(device=device)
        return net


"""
    分割模型中的参数
    named_parameters: 带名称的参数
    regex_expr: 正则表达式(r"")

    返回值：
        target, left
        target: 表示符合正则表达式的参数
        left: 表示不符合正则表达式的参数
"""


def split_parameters(named_parameters, regex_expr):
    target = []
    left = []

    pattern = re.compile(regex_expr)
    for name, param in named_parameters:
        if pattern.fullmatch(name):
            target.append((name, param))
        else:
            left.append((name, param))

    return target, left


"""
    获取优化器
    @:param net: 网络模型
    @:param optimizer_config: 优化器配置文件路径
    @:return 优化器
"""


def get_optimizer(
        net,
        optimizer_config=Path("config") / "optimizer.yaml"
):
    with optimizer_config.open("r", encoding="utf-8") as f:
        optimizer_config = yaml.load(f, Loader=yaml.FullLoader)

        base_config = optimizer_config["base_config"]
        lr = eval(base_config["kwargs"])["lr"]
        weight_decay = eval(base_config["kwargs"])["weight_decay"]

        parameters_config = optimizer_config["parameters"][1:]
        left = net.named_parameters()
        parameters = []

        for params_config in parameters_config[1:]:
            params, left = split_parameters(
                named_parameters=left,
                regex_expr=r'' + next(iter(params_config.values()))["regex_expr"]
            )
            params = list(
                map(
                    lambda tp: tp[-1], params
                )
            )
            parameters.append(params)

        parameters = [
            list(
                map(
                    lambda tp: tp[-1], left
                )
            ),
            *parameters
        ]
        params = [
            {
                'params': param,
                'lr': lr * next(iter(params_config.values())).setdefault('lr_mult', 1.0),
                'weight_decay': weight_decay * next(iter(params_config.values())).setdefault('weight_decay', 0.)
            }
            for idx, params_config in enumerate(parameters_config) for param in parameters[idx]
        ]

        optimizer = eval(f"optim.{base_config['optim_type']}")(params, **eval(base_config["kwargs"]))
    return optimizer


"""
    获取学习率调度器
    @:param optimizer: 优化器
    @:param lr_scheduler_config: 学习率调度器配置文件路径
    @:return 学习率调度器
"""


def get_lr_scheduler(
        optimizer,
        lr_scheduler_config=Path("config") / "lr_scheduler.yaml"
):
    lr_scheduler = None
    with lr_scheduler_config.open("r", encoding="utf-8") as f:
        lr_scheduler_config = yaml.load(f, yaml.FullLoader)
        lr_scheduler = learning_rate_scheduler.get_lr_scheduler(
            optimizer=optimizer,
            scheduler_type=eval(f"learning_rate_scheduler.SchedulerType.{lr_scheduler_config['scheduler_type']}"),
            kwargs=eval(lr_scheduler_config["kwargs"])
        )
    return lr_scheduler


"""
    搜寻模型权重文件和自己创建的模型中第一个不同的参数
    left: 元组，("模型名称": state_dict)
    right: 元组，("模型名称": state_dict)
    ignore_counts: 忽略不同的数目
    列表：
        {
            "row_num": 0,
            "模型名称1": "name1",
            "模型名称2": "name2"
        }
"""


def first_diff(left: tuple, right: tuple, ignore_counts=0):
    left = copy.deepcopy(left)
    left_name, left_state = left
    left_state = list(left_state.keys())
    left_ord = 0

    right = copy.deepcopy(right)
    right_name, right_state = right
    right_state = list(right_state.keys())
    right_ord = 0

    response = None

    while left_ord < len(left_state) and right_ord < len(right_state):
        left_sign = left_state[left_ord].split(".")[-1]
        right_sign = right_state[right_ord].split(".")[-1]
        print(f"{left_ord}: {left_state[left_ord]} --> {right_state[right_ord]}")
        if left_sign != right_sign:
            if ignore_counts != 0:
                ignore_counts -= 1
                left_ord += 1
                right_ord += 1
                continue

            assert left_ord == right_ord
            response = {
                "row_num": left_ord,
                left_name: left_state[left_ord],
                right_name: right_state[right_ord]
            }
            return response

        left_ord += 1
        right_ord += 1

    while ignore_counts:
        left_ord += 1
        right_ord += 1
        ignore_counts -= 1

    if left_ord < len(left_state) and right_ord >= len(right_state):
        response = {
            "row_num": left_ord,
            left_name: left_state[left_ord],
            right_name: "None"
        }
    if left_ord >= len(left_state) and right_ord < len(right_state):
        response = {
            "row_num": right_ord,
            left_name: "None",
            right_name: right_state[right_ord]
        }
    if left_ord >= len(left_state) and right_ord >= len(right_state):
        response = {
            "row_num": -1,
            left_name: "same",
            right_name: "same"
        }
    print(f"{response['row_num']}: {response[left_name]} --> {response[right_name]}")
    return response


"""
    初始化模型
    @:param train: 
        True表示，初始化用来训练的网络；
        False表示，初始化用来预测的网络.
    net: 网络模型
    optimizer: 优化器
    pretrained: 是否加载预训练权重
    @:param train_config: 训练配置文件路径
"""


def init_model(
        train,
        net,
        optimizer=None,
        train_config=Path("config") / "train.yaml",
        predict_config=Path("config") / "predict.yaml"
):
    # 初始化权重
    for m in net.modules():
        if isinstance(m, nn.Linear):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.LayerNorm):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.0)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            if m.weight is not None:
                nn.init.normal_(m.weight, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                nn.init.normal_(m.bias, 0.)

    if train:
        with train_config.open("r", encoding="utf-8") as tcf:
            config = yaml.load(tcf, yaml.FullLoader)
    else:
        with predict_config.open("r", encoding="utf-8") as pcf:
            config = yaml.load(pcf, yaml.FullLoader)

    mode = config["mode"]
    if mode == -1:
        return

    checkpoint = torch.load(os.path.sep.join(config["checkpoint"]))
    if mode == 0:
        for regex_expr in config["regex_expr"]:
            checkpoint["state_dict"] = {
                tp[0]: tp[-1]
                for tp in zip(net.state_dict().keys(), checkpoint["state_dict"].values())
                if re.compile(r"" + regex_expr).fullmatch(tp[0])
            }
        checkpoint["optimizer"]["state"] = dict()

    net.load_state_dict(checkpoint["state_dict"], strict=False)
    if train:
        optimizer.load_state_dict(checkpoint["optimizer"])


import json
import math

import torch.nn as nn
import torch
# import bricks
import torch.nn.functional as F
from abc import *

# import utils

"""
    [batch_size, in_channels, height, width] -> [batch_size, out_channels, height // 4, width // 4]
"""


class StemConv(nn.Module):

    def __init__(self, in_channels, out_channels, norm_layer=None):
        super(StemConv, self).__init__()

        self.proj = nn.Sequential(
            DownSampling(
                in_channels=in_channels,
                out_channels=out_channels // 2,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
            DownSampling(
                in_channels=out_channels // 2,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                norm_layer=norm_layer
            ),
        )

    def forward(self, x):
        out = self.proj(x)
        return out


class MSCA(nn.Module):

    def __init__(self, in_channels):
        super(MSCA, self).__init__()

        self.conv = DepthwiseConv(
            in_channels=in_channels,
            kernel_size=(5, 5),
            padding=(2, 2),
            bias=True
        )

        self.conv7 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 7),
                padding=(0, 3),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(7, 1),
                padding=(3, 0),
                bias=True
            )
        )

        self.conv11 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 11),
                padding=(0, 5),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(11, 1),
                padding=(5, 0),
                bias=True
            )
        )

        self.conv21 = nn.Sequential(
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(1, 21),
                padding=(0, 10),
                bias=True
            ),
            DepthwiseConv(
                in_channels=in_channels,
                kernel_size=(21, 1),
                padding=(10, 0),
                bias=True
            )
        )

        self.fc = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        u = x
        out = self.conv(x)

        branch1 = self.conv7(out)
        branch2 = self.conv11(out)
        branch3 = self.conv21(out)

        out = self.fc(out + branch1 + branch2 + branch3)
        out = out * u
        return out


class Attention(nn.Module):

    def __init__(self, in_channels):
        super(Attention, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )
        self.msca = MSCA(in_channels=in_channels)
        self.fc2 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=(1, 1)
        )

    def forward(self, x):
        out = F.gelu(self.fc1(x))
        out = self.msca(out)
        out = self.fc2(out)
        return out


class FFN(nn.Module):

    def __init__(self, in_features, hidden_features, out_features, drop_prob=0.):
        super(FFN, self).__init__()

        self.fc1 = nn.Conv2d(
            in_channels=in_features,
            out_channels=hidden_features,
            kernel_size=(1, 1)
        )
        self.dw = DepthwiseConv(
            in_channels=hidden_features,
            kernel_size=(3, 3),
            bias=True
        )
        self.fc2 = nn.Conv2d(
            in_channels=hidden_features,
            out_channels=out_features,
            kernel_size=(1, 1)
        )
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x):
        out = self.fc1(x)
        out = F.gelu(self.dw(out))
        out = self.fc2(out)
        out = self.dropout(out)
        return out


class Block(nn.Module):

    def __init__(self, in_channels, expand_ratio, drop_prob=0., drop_path_prob=0.):
        super(Block, self).__init__()

        self.norm1 = nn.BatchNorm2d(num_features=in_channels)
        self.attention = Attention(in_channels=in_channels)
        self.drop_path = DropPath(drop_prob=drop_path_prob if drop_path_prob >= 0 else nn.Identity)
        self.norm2 = nn.BatchNorm2d(num_features=in_channels)
        self.ffn = FFN(
            in_features=in_channels,
            hidden_features=int(expand_ratio * in_channels),
            out_features=in_channels,
            drop_prob=drop_prob
        )

        layer_scale_init_value = 1e-2
        self.layer_scale1 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )
        self.layer_scale2 = nn.Parameter(
            layer_scale_init_value * torch.ones(in_channels),
            requires_grad=True
        )

    def forward(self, x):
        out = self.norm1(x)
        out = self.attention(out)
        out = x + self.drop_path(
            self.layer_scale1.unsqueeze(-1).unsqueeze(-1) * out
        )
        x = out

        out = self.norm2(out)
        out = self.ffn(out)
        out = x + self.drop_path(
            self.layer_scale2.unsqueeze(-1).unsqueeze(-1) * out
        )

        return out


class Stage(nn.Module):

    def __init__(
            self,
            stage_id,
            in_channels,
            out_channels,
            expand_ratio,
            blocks_num,
            drop_prob=0.,
            drop_path_prob=[0.]
    ):
        super(Stage, self).__init__()

        assert blocks_num == len(drop_path_prob)

        if stage_id == 0:
            self.down_sampling = StemConv(
                in_channels=in_channels,
                out_channels=out_channels
            )
        else:
            self.down_sampling = DownSampling(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2)
            )

        self.blocks = nn.Sequential(
            *[
                Block(
                    in_channels=out_channels,
                    expand_ratio=expand_ratio,
                    drop_prob=drop_prob,
                    drop_path_prob=drop_path_prob[i]
                ) for i in range(0, blocks_num)
            ]
        )

        self.norm = nn.LayerNorm(out_channels)

    def forward(self, x):
        out = self.down_sampling(x)
        out = self.blocks(out)
        # [batch_size, channels, height, width] -> [batch_size, channels, height * width]
        batch_size, channels, height, width = out.shape
        out = out.view(batch_size, channels, -1)
        # [batch_size, channels, height * width] -> [batch_size, height * width, channels]
        out = torch.transpose(out, -2, -1)
        out = self.norm(out)

        # [batch_size, height * width, channels] -> [batch_size, channels, height * width]
        out = torch.transpose(out, -2, -1)
        # [batch_size, channels, height * width] -> [batch_size, channels, height, width]
        out = out.view(batch_size, -1, height, width)

        return out


class MSCAN(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],
            expand_ratios=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob=0.1,
            drop_path_prob=0.1
    ):
        super(MSCAN, self).__init__()

        dpr = [x.item() for x in torch.linspace(0, drop_path_prob, sum(depths))]
        self.stages = nn.Sequential(
            *[
                Stage(
                    stage_id=stage_id,
                    in_channels=embed_dims[stage_id],
                    out_channels=embed_dims[stage_id + 1],
                    expand_ratio=expand_ratios[stage_id],
                    blocks_num=depths[stage_id],
                    drop_prob=drop_prob,
                    drop_path_prob=dpr[sum(depths[: stage_id]): sum(depths[: stage_id + 1])]
                ) for stage_id in range(0, len(depths))
            ]
        )

    def forward(self, x):
        out = x
        outputs = []

        for idx, stage in enumerate(self.stages):
            out = stage(out)
            if idx != 0:
                outputs.append(out)

        # outputs: [output_of_stage1, output_of_stage2, output_of_stage3]
        # output_of_stage1: [batch_size, embed_dims[2], height / 8, width / 8]
        # output_of_stage2: [batch_size, embed_dims[3], height / 16, width / 16]
        # output_of_stage3: [batch_size, embed_dims[4], height / 32, width / 32]
        return [x, *outputs]


class Hamburger(nn.Module):

    def __init__(
            self,
            hamburger_channels=256,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(Hamburger, self).__init__()
        self.ham_in = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1)
            )
        )

        self.ham = NMF2D(args=nmf2d_config)

        self.ham_out = nn.Sequential(
            nn.Conv2d(
                in_channels=hamburger_channels,
                out_channels=hamburger_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hamburger_channels
            )
        )

    def forward(self, x):
        out = self.ham_in(x)
        out = self.ham(out)
        out = self.ham_out(out)
        out = F.relu(x + out)
        return out


class LightHamHead(nn.Module):

    def __init__(
            self,
            in_channels_list=[64, 160, 256],
            hidden_channels=256,
            out_channels=256,
            classes_num=150,
            drop_prob=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": True,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(LightHamHead, self).__init__()

        self.cls_seg = nn.Sequential(
            nn.Dropout2d(drop_prob),
            nn.Conv2d(
                in_channels=256,
                out_channels=classes_num,
                kernel_size=(1, 1)
            )
        )

        self.squeeze = nn.Sequential(
            nn.Conv2d(
                in_channels=sum(in_channels_list),
                out_channels=hidden_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=hidden_channels,
            ),
            nn.ReLU()
        )

        self.hamburger = Hamburger(
            hamburger_channels=hidden_channels,
            nmf2d_config=nmf2d_config
        )

        self.align = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                bias=False
            ),
            nn.GroupNorm(
                num_groups=32,
                num_channels=out_channels
            ),
            nn.ReLU()
        )

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1))
        self.conv3 = nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(1, 1))



    # inputs: [x, x_1, x_2, x_3]
    # x: [batch_size, channels, height, width]
    def forward(self, inputs):
        assert len(inputs) >= 2
        o = inputs[0]
        batch_size, _, standard_height, standard_width = inputs[1].shape
        standard_shape = (standard_height, standard_width)
        inputs = [
            F.interpolate(
                input=x,
                size=standard_shape,
                mode="bilinear",
                align_corners=False
            )
            for x in inputs[1:]
        ]

        # x: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        x = torch.cat(inputs, dim=1)

        # out: [batch_size, channels_1 + channels_2 + channels_3, standard_height, standard_width]
        out = self.squeeze(x)
        out = self.hamburger(out)
        out = self.align(out)  # (N, 256, 64, 64)

        # out: [batch_size, classes_num, standard_height, standard_width]
        out = self.cls_seg(out)  # (N, num_classes, 64, 64)
        # print(out.shape)

        out = F.interpolate(input=out, size=(128, 128), mode="bilinear", align_corners=False)
        out = self.conv1(out)

        out = F.interpolate(input=out, size=(512, 512), mode="bilinear", align_corners=False)
        out = self.conv2(out)

        _, _, original_height, original_width = o.shape
        # out: [batch_size, original_height * original_width, classes_num]
        # out = F.interpolate(
        #     input=out,
        #     size=(original_height, original_width),
        #     mode="bilinear",
        #     align_corners=False
        # )
        # out = torch.transpose(out.view(batch_size, -1, original_height * original_width), -2, -1)

        return out


class SegNeXt(nn.Module):

    def __init__(
            self,
            embed_dims=[3, 32, 64, 160, 256],
            expand_rations=[8, 8, 4, 4],
            depths=[3, 3, 5, 2],
            drop_prob_of_encoder=0.1,
            drop_path_prob=0.1,
            hidden_channels=256,
            out_channels=256,
            classes_num=150,
            drop_prob_of_decoder=0.1,
            nmf2d_config=json.dumps(
                {
                    "SPATIAL": True,
                    "MD_S": 1,
                    "MD_D": 512,
                    "MD_R": 64,
                    "TRAIN_STEPS": 6,
                    "EVAL_STEPS": 7,
                    "INV_T": 1,
                    "ETA": 0.9,
                    "RAND_INIT": False,
                    "return_bases": False,
                    "device": "cuda"
                }
            )
    ):
        super(SegNeXt, self).__init__()

        self.encoder = MSCAN(
            embed_dims=embed_dims,
            expand_ratios=expand_rations,
            depths=depths,
            drop_prob=drop_prob_of_encoder,
            drop_path_prob=drop_path_prob
        )

        self.decoder = LightHamHead(
            in_channels_list=embed_dims[-3:],
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            classes_num=classes_num,
            drop_prob=drop_prob_of_decoder,
            nmf2d_config=nmf2d_config
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class SegNext(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = SegNeXt(classes_num=3)

    def forward(self, image):
        out = self.model(image)

        return out