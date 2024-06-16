import torch.nn as nn


class IntermediateSequential(nn.Sequential):
    def __init__(self, *args, return_intermediate=True):
        super().__init__(*args)
        self.return_intermediate = return_intermediate

    def forward(self, input):
        if not self.return_intermediate:
            return super().forward(input)

        intermediate_outputs = {}
        output = input
        for name, module in self.named_children():
            output = intermediate_outputs[name] = module(output)

        return output, intermediate_outputs

import torch.nn as nn
#from model.IntmdSequential import IntermediateSequential


class SelfAttention(nn.Module):
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)

        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn

    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )

    def forward(self, x):
        return self.net(x)


class TransformerModel(nn.Module):
    def __init__(
        self,
        dim,   #embedding_dim
        depth,
        heads,
        mlp_dim,
        dropout_rate=0.1,
        attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate),
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )

        self.net = IntermediateSequential(*layers)


    def forward(self, x):
        return self.net(x)


import torch
from torchvision import models as resnet_model
from torch import nn
#from model.transformer import TransformerModel

class SEBlock(nn.Module):
    def __init__(self, channel, r=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // r, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze
        y = self.avg_pool(x).view(b, c)
        # Excitation
        y = self.fc(y).view(b, c, 1, 1)
        # Fusion
        y = torch.mul(x, y)
        return y

class DecoderBottleneckLayer(nn.Module):
    def __init__(self, in_channels):
        super(DecoderBottleneckLayer, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.norm3 = nn.BatchNorm2d(in_channels)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ParaTransCNN(nn.Module):
    def __init__(self, n_channels=3, num_classes=9, heads=8, dim=320, depth=(3, 3, 3), patch_size=2):
        super(ParaTransCNN, self).__init__()
        self.num_classes = num_classes
        self.n_channels = n_channels
        self.patch_size = patch_size
        self.heads = heads
        self.depth = depth
        self.dim = dim
        mlp_dim = [2 * dim, 4 * dim, 8 * dim, 16 * dim]
        embed_dim = [dim, 2 * dim, 4 * dim, 8 * dim]
        resnet = resnet_model.resnet34(weights=resnet_model.ResNet34_Weights.DEFAULT) # pretrained = True
        self.vit_1 = TransformerModel(dim=embed_dim[0], mlp_dim=mlp_dim[0],depth=depth[0], heads=heads)
        self.vit_2 = TransformerModel(dim=embed_dim[1], mlp_dim=mlp_dim[1],depth=depth[1], heads=heads)
        self.vit_3 = TransformerModel(dim=embed_dim[2], mlp_dim=mlp_dim[2],depth=depth[2], heads=heads)
        self.patch_embed_1 = nn.Conv2d(n_channels,embed_dim[0],kernel_size=2*patch_size,stride=2*patch_size)
        self.patch_embed_2 = nn.Conv2d(embed_dim[0], embed_dim[1], kernel_size=patch_size, stride=patch_size)
        self.patch_embed_3 = nn.Conv2d(embed_dim[1], embed_dim[2], kernel_size=patch_size, stride=patch_size)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.SE_1 = SEBlock(4*dim + 512)
        self.SE_2 = SEBlock(2*dim + 256)
        self.SE_3 = SEBlock(dim + 128)
        self.decoder1 = DecoderBottleneckLayer(4*dim + 512)
        self.decoder2 = DecoderBottleneckLayer(4*dim + 512)
        self.decoder3 = DecoderBottleneckLayer(dim + 128 + 2*dim + 256)
        self.up3_1 = nn.ConvTranspose2d(4*dim + 512, 2*dim + 256, kernel_size=2, stride=2)
        self.up2_1 = nn.ConvTranspose2d(4*dim + 512, 2*dim + 256, kernel_size=2, stride=2)
        self.up1_1 = nn.ConvTranspose2d(dim + 128 + 2*dim + 256, dim, kernel_size=4, stride=4)
        self.out = nn.Conv2d(dim, num_classes,kernel_size=1)

    def forward(self, x):
        b, c, h, w = x.shape
        patch_size = self.patch_size
        dim = self.dim
        e0 = self.firstconv(x)
        e0 = self.firstbn(e0)
        e0 = self.firstrelu(e0)

        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        v1 = self.patch_embed_1(x)
        v1 = v1.permute(0, 2, 3, 1).contiguous()
        v1 = v1.view(b, -1, dim)
        v1, v6 = self.vit_1(v1)
        v1_cnn = v1.view(b, int(h / (2*patch_size)), int(w / (2*patch_size)), dim)
        v1_cnn = v1_cnn.permute(0, 3, 1, 2).contiguous()

        v2 = self.patch_embed_2(v1_cnn)
        v2 = v2.permute(0, 2, 3, 1).contiguous()
        v2 = v2.view(b, -1, 2*dim)
        v2, _ = self.vit_2(v2)
        v2_cnn = v2.view(b, int(h / (patch_size*2*2)), int(w / (2*2*patch_size)), dim*2)
        v2_cnn = v2_cnn.permute(0, 3, 1, 2).contiguous()

        v3 = self.patch_embed_3(v2_cnn)
        v3 = v3.permute(0, 2, 3, 1).contiguous()
        v3 = v3.view(b, -1, 4*dim)
        v3, _ = self.vit_3(v3)
        v3_cnn = v3.view(b, int(h / (patch_size * 2*2*2)), int(w / (2*2*2 * patch_size)), dim*2 * 2)
        v3_cnn = v3_cnn.permute(0, 3, 1, 2).contiguous()

        cat_1 = torch.cat([v3_cnn, e4], dim=1)
        cat_1 = self.SE_1(cat_1)
        cat_1 = self.decoder1(cat_1)
        cat_1 = self.up3_1(cat_1)

        cat_2 = torch.cat([v2_cnn, e3], dim=1)
        cat_2 = self.SE_2(cat_2)
        cat_2 = torch.cat([cat_2, cat_1],dim=1)
        cat_2 = self.decoder2(cat_2)
        cat_2 = self.up2_1(cat_2)

        cat_3 = torch.cat([v1_cnn, e2], dim=1)
        cat_3 = self.SE_3(cat_3)
        cat_3 = torch.cat([cat_3, cat_2], dim=1)
        cat_3 = self.decoder3(cat_3)
        cat_3 = self.up1_1(cat_3)
        out = self.out(cat_3)

        return out