import torch.nn as nn
import math


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet50(nn.Module):
    def __init__(self):
        self.inplanes = 64
        super(ResNet50, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(Bottleneck, 64, 3)
        self.layer2 = self._make_layer(Bottleneck, 128, 4, stride=2)
        self.layer3 = self._make_layer(Bottleneck, 256, 6, stride=2)
        self.layer4 = self._make_layer(Bottleneck, 512, 3, stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.conv1(x)    # 1/2
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x2 = self.maxpool(x1)  # 1/4

        x2 = self.layer1(x2)   # 1/4
        x3 = self.layer2(x2)   # 1/8
        x4 = self.layer3(x3)   # 1/16
        x5 = self.layer4(x4)   # 1/32

        return x1, x2, x3, x4, x5


import torch
import torch.nn as nn
import torch.nn.functional as F
# from ResNet50 import ResNet50
import torchvision.models as models


class MSCM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(MSCM, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.branch1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=2, dilation=2), nn.ReLU(True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 5, padding=2, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=4, dilation=4), nn.ReLU(True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 7, padding=3, dilation=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=6, dilation=6), nn.ReLU(True),
        )
        self.score = nn.Conv2d(out_channel * 4, 1, 3, padding=1)

    def forward(self, x):
        x = self.convert(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x4 = self.branch4(x)

        x = torch.cat((x1, x2, x3, x4), 1)
        x = self.score(x)

        return x


class RA(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RA, self).__init__()
        self.convert = nn.Conv2d(in_channel, out_channel, 1)
        self.convs = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, out_channel, 3, padding=1), nn.ReLU(True),
            nn.Conv2d(out_channel, 1, 3, padding=1),
        )
        self.channel = out_channel

    def forward(self, x, y):
        '''x: encoder feature - (N, C, H, W)
           y: current binary seg prediction (upsampled) - (N, 1, H, W) '''
        a = torch.sigmoid(-y)  # reverse-attention weight

        x = self.convert(x)  # (N, C, H, W) -> (N, out_channel, H, W)
        x = a.expand(-1, self.channel, -1, -1).mul(x)  # weighted conv feature

        # self.convs(x): side-output residual feature
        # refine current binary seg prediction
        y = y + self.convs(x)

        return y


class RAS(nn.Module):
    def __init__(self, channel=64):
        super(RAS, self).__init__()
        self.resnet = ResNet50()
        self.mscm = MSCM(2048, channel)

        self.ra1 = RA(64, channel)
        self.ra2 = RA(256, channel)
        self.ra3 = RA(512, channel)
        self.ra4 = RA(1024, channel)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.initialize_weights()

    def forward(self, x):
        '''x: (N, 3, 512, 512)'''

        # ---------- Encoder features ----------
        # x1: (N, 64, 256, 256)
        # x2: (N, 256, 128, 128)
        # x3: (N, 512, 64, 64)
        # x4: (N, 1024, 32, 32)
        # x5: (N, 2048, 16, 16)
        x1, x2, x3, x4, x5 = self.resnet(x)

        x_size = x.size()[2:]
        x1_size = x1.size()[2:]
        x2_size = x2.size()[2:]
        x3_size = x3.size()[2:]
        x4_size = x4.size()[2:]

        y5 = self.mscm(x5)  # (N, 2048, 16, 16) -> (N, 1, 16, 16) - Global saliency
        score5 = F.interpolate(y5, x_size, mode='bilinear', align_corners=True)  # (N, 1, 512, 512)

        # --------- reverse-attention 1 ---------
        y5_4 = F.interpolate(y5, x4_size, mode='bilinear',
                             align_corners=True)  # (N, 1, 32, 32) - upx2 current prediction y5
        y4 = self.ra4(x4, y5_4)  # (N, 1, 32, 32)
        score4 = F.interpolate(y4, x_size, mode='bilinear', align_corners=True)  # (N, 1, 512, 512)

        # --------- reverse-attention 2 ---------
        y4_3 = F.interpolate(y4, x3_size, mode='bilinear',
                             align_corners=True)  # (N, 1, 64, 64) - upx2 current prediction y4
        y3 = self.ra3(x3, y4_3)  # (N, 1, 64, 64)
        score3 = F.interpolate(y3, x_size, mode='bilinear', align_corners=True)  # (N, 1, 512, 512)

        # --------- reverse-attention 3 ---------
        y3_2 = F.interpolate(y3, x2_size, mode='bilinear',
                             align_corners=True)  # (N, 1, 128, 128) - upx2 current prediction y3
        y2 = self.ra2(x2, y3_2)  # (N, 1, 128, 128)
        score2 = F.interpolate(y2, x_size, mode='bilinear', align_corners=True)  # (N, 1, 512, 512)

        # --------- reverse-attention 4 ---------
        y2_1 = F.interpolate(y2, x1_size, mode='bilinear',
                             align_corners=True)  # (N, 1, 256, 256) - upx2 current prediction y2
        y1 = self.ra1(x1, y2_1)  # (N, 1, 256, 256)
        score1 = F.interpolate(y1, x_size, mode='bilinear', align_corners=True)  # (N, 1, 512, 512)

        return score1, score2, score3, score4, score5

    def initialize_weights(self):
        res50 = models.resnet50(pretrained=False)
        self.resnet.load_state_dict(res50.state_dict(), False)
