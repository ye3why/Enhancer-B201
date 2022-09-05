#!/usr/bin/env python
import math
import numpy as np
import torch.nn.init as init
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from registry import MODEL_REGISTRY

# RFA是 30*4*2层卷积 另有注意力机制
# RCAN是 10*20*2层卷积，更大
class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super(Encoder, self).__init__()
        self.head = ConvNorm(in_channels, 32, kernel_size=1)
        self.extractfeats= RIR(5, 32)

    def forward(self, x):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats = self.head(x)
        feats = self.extractfeats(feats)

        return feats

class CAR(nn.Module):
    def __init__(self, in_channels=3):
        super(CAR, self).__init__()

        self.lrelu = nn.LeakyReLU(0.01)

        head = []
        head.append(ConvNorm(160, 32, kernel_size=1, stride=1))
        self.head = nn.Sequential(*head)

        self.fusion1 = RIR(6, 32)
        self.fusion2 = RIR(6, 32)

        rfa = []
        rfa.append(ConvNorm(32 * 3, 64))
        rfa.append(self.lrelu)
        rfa.append(ConvNorm(64, 32))
        self.rfa = nn.Sequential(*rfa)

        self.tail = ConvNorm(32, 3)

    def forward(self, x):
        """
        Encoder: Shuffle-spread --> Feature Fusion --> Return fused features
        """
        feats = self.head(x)
        feats_ori = feats
        feats1 = self.fusion1(feats)
        feats2 = self.fusion2(feats1)
        # B, C+C+C, H, W
        feats = torch.cat([feats_ori, feats1, feats2], 1)
        feats = self.rfa(feats) # add rfa

        out = self.tail(feats)

        return out

@MODEL_REGISTRY.register()
class ECAR(nn.Module):
    def __init__(self):
        super(ECAR, self).__init__()

        self.enc = Encoder(in_channels=3).to('cuda:0')
        self.car = CAR(in_channels=3).to('cuda:1')


    def forward(self, x):
        x.to('cuda:0')
        B, T, C, H, W = x.size()
        x_center = x[:, T//2, :, :, :]

        base = x_center

        x = x.reshape(-1, C, H, W)

        x = self.enc(x)
        x.to('cuda:1')

        x_car = x.reshape(B, -1, H, W)

        x_car = self.car(x_car)

        out = x_car + base

        return out

class ConvNorm(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size=3, stride=1):
        super(ConvNorm, self).__init__()

        padding = kernel_size // 2
        self.conv = nn.Conv2d(in_feat, out_feat, stride=stride, kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        out = self.conv(x)
        return out


class ESA(nn.Module):
    def __init__(self, n_feats):
        super(ESA, self).__init__()
        f = n_feats // 4
        self.conv1 = nn.Conv2d(n_feats, f, kernel_size=1)
        self.conv_f = nn.Conv2d(f, f, kernel_size=1)
        self.conv_max = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(f, f, kernel_size=3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv3_ = nn.Conv2d(f, f, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(f, n_feats, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    # 和官方实现有歧义
    def forward(self, x):
        c1_ = (self.conv1(x))
        c1 = self.conv2(c1_)
        # v_max = self.pooling(c1)
        v_max = F.max_pool2d(c1, kernel_size=7, stride=3)
        v_range = self.relu(self.conv_max(v_max))
        c3 = self.relu(self.conv3(v_range))
        c3 = self.conv3_(c3)
        c3 = F.interpolate(c3, (x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        cf = self.conv_f(c1_)
        c4 = self.conv4(c3 + cf)
        m = self.sigmoid(c4)

        return x * m, m

""" CONV - (BN) - RELU - CONV - (BN) """


## Channel Attention (CA) Layer

## Residual Block with ESA


class HDC(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, n_feat):
        super(HDC, self).__init__()

        self.branch1 = nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)

        self.branch2 = nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=2, bias=True, dilation=2)

        self.branch3 = nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)

        self.branch5 = nn.Conv2d(n_feat, n_feat//4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)

        self.ConvLinear = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.1, False)

        self.sa = ESA(n_feat)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch5 = self.branch5(x)

        outputs = [branch1, branch2, branch3, branch5]
        return outputs

    def forward(self, x):
        res = x
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear(outputs))

        outputs, mask = self.sa(outputs)
        rf = outputs
        outputs += res
        return outputs, rf


class EHDC(nn.Module):
    __constants__ = ['branch1', 'branch2', 'branch3', 'branch5']

    def __init__(self, n_feat, act):
        super(EHDC, self).__init__()

        self.act = act

        self.branch1 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=3, stride=1, padding=1, bias=True, dilation=1)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//4, kernel_size=1, stride=1),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=3, stride=1, padding=3, bias=True, dilation=3)
        )

        self.branch4 = nn.Sequential(
            nn.Conv2d(n_feat, n_feat//8, kernel_size=1, stride=1),
            act,
            nn.Conv2d(n_feat//8, n_feat//8 + 2, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            act,
            nn.Conv2d(n_feat//8+2, n_feat//4, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            act,
            nn.Conv2d(n_feat//4, n_feat//4, kernel_size=3, stride=1, padding=5, bias=True, dilation=5)
        )

        self.ConvLinear = nn.Conv2d(n_feat, n_feat, kernel_size=1, stride=1)
        self.lrelu = nn.LeakyReLU(0.1, False)

        self.sa = ESA(n_feat)

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        res = x
        outputs = self._forward(x)
        outputs = torch.cat(outputs, 1)
        outputs = self.lrelu(self.ConvLinear(outputs))

        outputs, mask = self.sa(outputs)

        rf = outputs
        outputs += res

        return outputs, rf


class RB_last(nn.Module):
    def __init__(self, in_feat, out_feat, kernel_size, act=nn.LeakyReLU(0.01)):
        super(RB_last, self).__init__()

        self.body = nn.Sequential(
            ConvNorm(in_feat, out_feat, kernel_size=kernel_size, stride=1),
            act,
            ConvNorm(out_feat, out_feat, kernel_size=kernel_size, stride=1),
            ESA(out_feat)
        )

    def forward(self, x):
        # res = x
        out, mask = self.body(x)
        # out += res
        return out


## Residual Group (RG)

class BM(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(BM, self).__init__()

        self.HDC1 = HDC(n_feat)

        self.HDC2 = HDC(n_feat)

        self.HDC3 = HDC(n_feat)

        self.RB4 = RB_last(n_feat, n_feat, kernel_size)

        self.Aggre = ConvNorm(n_feat * 4, n_feat, kernel_size=1, stride=1)

    def forward(self, x):
        out, rf1 = self.HDC1(x)
        out, rf2 = self.HDC2(out)
        out, rf3 = self.HDC3(out)

        out = self.RB4(out)
        # print(out.shape,rf1.shape, rf2.shape, rf3.shape)
        # B, C, H, W
        out = torch.cat([out, rf1, rf2, rf3], 1)
        out = self.Aggre(out)
        out += x
        return out


class EBM(nn.Module):
    def __init__(self, n_feat, kernel_size):
        super(EBM, self).__init__()

        self.act = nn.ReLU()
        self.EHDC1 = EHDC(n_feat, self.act)

        self.EHDC2 = EHDC(n_feat, self.act)

        self.EHDC3 = EHDC(n_feat, self.act)

        self.RB4 = RB_last(n_feat, n_feat, kernel_size)

        self.Aggre = ConvNorm(n_feat * 4, n_feat, kernel_size=1, stride=1)

    def forward(self, x):
        out, rf1 = self.EHDC1(x)
        out, rf2 = self.EHDC2(out)
        out, rf3 = self.EHDC3(out)

        out = self.RB4(out)
        # B, C, H, W
        # print(out.shape, rf1.shape, rf2.shape, rf3.shape)
        out = torch.cat([out, rf1, rf2, rf3], 1)
        out = self.Aggre(out)
        out += x
        return out


def pixel_shuffle(input, scale_factor):
    batch_size, channels, in_height, in_width = input.shape

    out_channels = int(int(channels / scale_factor) / scale_factor)
    out_height = int(in_height * scale_factor)
    out_width = int(in_width * scale_factor)

    if scale_factor >= 1:
        input_view = input.reshape(batch_size, out_channels, scale_factor, scale_factor, in_height, in_width)
        # shuffle_out = F.dimshuffle(input_view, (0, 1, 4, 3, 5, 2))
        shuffle_out = input_view.permute(0, 1, 4, 2, 5, 3).contiguous()
        # B C H r W r  通道↓ 尺寸↑

    return shuffle_out.reshape(batch_size, out_channels, out_height, out_width)


class PixelShuffle(nn.Module):
    def __init__(self, scale_factor):
        super(PixelShuffle, self).__init__()
        self.scale_factor = scale_factor  # scale_factor = 4
        self.conv = ConvNorm(64, int(64 * (self.scale_factor ** 2)))  # 64 → 1024

    def forward(self, x):
        # as RCAN do , pixel shuffle n_feats2n_feats
        if self.scale_factor >= 1:
            x = self.conv(x)
        return pixel_shuffle(x, self.scale_factor)


# RIR : n_basemodules = 30
class RIR(nn.Module):
    def __init__(self, n_basemodules, n_feats):
        super(RIR, self).__init__()

        self.headConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)
        # define modules: head, body, tail
        modules_body = [
            BM(
                n_feat=n_feats,
                kernel_size=3)
            for _ in range(n_basemodules)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)

    def forward(self, x):
        # Build input tensor B, C, H, W
        x = self.headConv(x)
        res = self.body(x)
        res += x
        out = self.tailConv(res)
        return out

class ERIR(nn.Module):
    def __init__(self, n_basemodules, n_feats):
        super(ERIR, self).__init__()

        self.headConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)
        # define modules: head, body, tail
        modules_body = [
            EBM(
                n_feat=n_feats,
                kernel_size=3)
            for _ in range(n_basemodules)]
        self.body = nn.Sequential(*modules_body)

        self.tailConv = ConvNorm(n_feats, n_feats, kernel_size=3, stride=1)

    def forward(self, x):
        # Build input tensor B, C, H, W
        x = self.headConv(x)
        res = self.body(x)
        res += x
        out = self.tailConv(res)
        return out
