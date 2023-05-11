import torch.nn as nn
import functools
import torch.nn.functional as F
import torch
import math
from registry import MODEL_REGISTRY

def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.size()

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.contiguous().view(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
    else:
        block_size = int(1 / scale_factor)
        input_view = input.contiguous().view(batch_size, channels, out_height, block_size, out_width, block_size)
        shuffle_out = input_view.permute(0, 1, 3, 5, 2, 4).contiguous()

    return shuffle_out.view(batch_size, out_channels, out_height, out_width)


class PixelShuffleD(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffleD, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, x):
        return pixel_shuffle(x, self.scale_factor)
    def extra_repr(self):
        return 'scale_factor={}'.format(self.scale_factor)

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class RCAB(nn.Module):
    def __init__(
        self, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = []
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class RFA(nn.Module):
    def __init__(self, n_feat):
        super(RFA, self).__init__()
        self.rb1 = RCAB(default_conv, n_feat, 3, reduction=16)
        self.rb2 = RCAB(default_conv, n_feat, 3, reduction=16)
        self.rb3 = RCAB(default_conv, n_feat, 3, reduction=16)
        self.rb4 = RCAB(default_conv, n_feat, 3, reduction=16)
        self.aggr_conv = nn.Conv2d(n_feat * 4, n_feat, 1)


    def forward(self, x):
        rb1_out = self.rb1(x)
        res = rb1_out + x
        rb2_out = self.rb2(res)
        res = rb2_out + res
        rb3_out = self.rb3(res)
        res = rb3_out + res
        rb4_out = self.rb4(res)
        res = self.aggr_conv(torch.cat([rb1_out, rb2_out, rb3_out, rb4_out], dim=1))
        return res + x


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)


@MODEL_REGISTRY.register()
class DescratchNet(nn.Module):
    def __init__(self, n_feats=32, n_frames=3):
        super(DescratchNet, self).__init__()

        n_rfas = 3

        self.conv_first = nn.Conv2d(3 * n_frames * 4, n_feats, 3, 1, 1)
        self.downshuffle = PixelShuffleD(1 / 2)
        self.upshuffle = PixelShuffleD(2)
        self.rfas = nn.Sequential(*[
            RFA(n_feats) for _ in range(n_rfas)
        ])
        self.conv_last = nn.Conv2d(n_feats // 4, 3, 3, 1, 1)


    def forward(self, x):
        B, N, C, H, W = x.size()
        x_center = x[:, N // 2]
        out = self.downshuffle(x.view(-1, C, H, W))
        out = out.view(B, -1, H // 2, W // 2)
        out = self.conv_first(out)
        out = self.rfas(out)
        out = self.upshuffle(out)
        out = self.conv_last(out)
        return out + x_center

