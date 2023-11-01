import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class InvRescaleNet(nn.Module):
    def __init__(self, channel_in=3, channel_out=3, subnet_type=None, block_num=[], down_first=False, use_ConvDownsampling=False, down_scale=4):
        super(InvRescaleNet, self).__init__()
        subnet_constructor = subnet(subnet_type, gc=32)
        down_num = int(math.log(down_scale, 2))

        operations = []
        self.down_scale = down_scale

        if use_ConvDownsampling:
            down_num = 1
            down_first = True

        current_channel = channel_in
        if down_first:
            for i in range(down_num):
                if use_ConvDownsampling:
                    b = ConvDownsampling(down_scale)
                    current_channel *= down_scale**2
                else:
                    b = HaarDownsampling(current_channel)
                    current_channel *= 4
                operations.append(b)
            for j in range(block_num[0]):
                b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                operations.append(b)
        else:
            for i in range(down_num):
                b = HaarDownsampling(current_channel)
                operations.append(b)
                current_channel *= 4
                for j in range(block_num[i]):
                    b = InvBlockExp(subnet_constructor, current_channel, channel_out)
                    operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward_orig(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

    def gaussian_batch(self, dims, x):
        return torch.randn(tuple(dims)).to(x.device)

    def forward(self, x):
        Lshape = x.shape

        input_dim = Lshape[1]
        zshape = [Lshape[0], input_dim * (self.down_scale**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        y_forw = torch.cat((x, gaussian_scale * self.gaussian_batch(zshape, x)), dim=1)
        fake_H = self.forward_orig(x=y_forw, rev=True)[:, :3, :, :]
        return fake_H

@MODEL_REGISTRY.register()
class InvRescaleNet_HaarDown(InvRescaleNet):
    def __init__(self, channel_in=3, channel_out=3, subnet_type=None, block_num=[], down_first=False, use_ConvDownsampling=False, down_scale=4):
        super(InvRescaleNet_HaarDown, self).__init__(channel_in=channel_in, channel_out=channel_out, subnet_type=subnet_type, block_num=block_num, down_first=down_first, use_ConvDownsampling=use_ConvDownsampling, down_scale=down_scale)
        # self.downsample = HaarDownsampling(channel_in)
        self.channel_in = channel_in
        self.down_num = int(math.log(down_scale, 2))

    def forward(self, x):
        for i in range(self.down_num):
            downsample = HaarDownsampling(self.channel_in).to(x.device)
            x = downsample(x)[:, :3]
        Lshape = x.shape

        input_dim = Lshape[1]
        zshape = [Lshape[0], input_dim * (self.down_scale**2) - Lshape[1], Lshape[2], Lshape[3]]

        gaussian_scale = 1
        y_forw = torch.cat((x, gaussian_scale * self.gaussian_batch(zshape, x)), dim=1)
        fake_H = self.forward_orig(x=y_forw, rev=True)[:, :3, :, :]
        return fake_H

class Quant(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        input = torch.clamp(input, 0, 1)
        output = (input * 255.).round() / 255.
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Quantization(nn.Module):
    def __init__(self):
        super(Quantization, self).__init__()

    def forward(self, input):
        return Quant.apply(input)

@MODEL_REGISTRY.register()
class InvRescaleNet_full(InvRescaleNet):
    def __init__(self, channel_in=3, channel_out=3, subnet_type=None, block_num=[], down_first=False, use_ConvDownsampling=False, down_scale=4):
        super(InvRescaleNet_full, self).__init__(channel_in=channel_in, channel_out=channel_out, subnet_type=subnet_type, block_num=block_num, down_first=down_first, use_ConvDownsampling=use_ConvDownsampling, down_scale=down_scale)
        self.Quantization = Quantization()

    def forward(self, x):
        Lshape = x.shape

        input_dim = Lshape[1]
        zshape = [Lshape[0], input_dim * (self.down_scale**2) - Lshape[1], Lshape[2], Lshape[3]]

        forw_L = self.forward_orig(x=x)[:, :3, :, :]
        forw_L = self.Quantization(forw_L)
        gaussian_scale = 1
        y_forw = torch.cat((x, gaussian_scale * self.gaussian_batch(zshape, x)), dim=1)
        fake_H = self.forward_orig(x=y_forw, rev=True)[:, :3, :, :]
        return fake_H



class DenseBlock(nn.Module):
    def __init__(self, channel_in, channel_out, init='xavier', gc=32, bias=True):
        super(DenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(channel_in, gc, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(channel_in + gc, gc, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(channel_in + 2 * gc, gc, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(channel_in + 3 * gc, gc, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(channel_in + 4 * gc, channel_out, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)


    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))

        return x5


def subnet(net_structure, init='xavier', gc=32):
    def constructor(channel_in, channel_out):
        if net_structure == 'DBNet':
            if init == 'xavier':
                return DenseBlock(channel_in, channel_out, init, gc=gc)
            else:
                return DenseBlock(channel_in, channel_out, gc=gc)
        else:
            return None

    return constructor


class InvBlockExp(nn.Module):
    def __init__(self, subnet_constructor, channel_num, channel_split_num, clamp=1.):
        super(InvBlockExp, self).__init__()

        self.split_len1 = channel_split_num
        self.split_len2 = channel_num - channel_split_num

        self.clamp = clamp

        self.F = subnet_constructor(self.split_len2, self.split_len1)
        self.G = subnet_constructor(self.split_len1, self.split_len2)
        self.H = subnet_constructor(self.split_len1, self.split_len2)

    def forward(self, x, rev=False):
        x1, x2 = (x.narrow(1, 0, self.split_len1), x.narrow(1, self.split_len1, self.split_len2))

        if not rev:
            y1 = x1 + self.F(x2)
            self.s = self.clamp * (torch.sigmoid(self.H(y1)) * 2 - 1)
            y2 = x2.mul(torch.exp(self.s)) + self.G(y1)
        else:
            self.s = self.clamp * (torch.sigmoid(self.H(x1)) * 2 - 1)
            y2 = (x2 - self.G(x1)).div(torch.exp(self.s))
            y1 = x1 - self.F(y2)

        return torch.cat((y1, y2), 1)

    def jacobian(self, x, rev=False):
        if not rev:
            jac = torch.sum(self.s)
        else:
            jac = -torch.sum(self.s)

        return jac / x.shape[0]


class HaarDownsampling(nn.Module):
    def __init__(self, channel_in):
        super(HaarDownsampling, self).__init__()
        self.channel_in = channel_in

        self.haar_weights = torch.ones(4, 1, 2, 2)

        self.haar_weights[1, 0, 0, 1] = -1
        self.haar_weights[1, 0, 1, 1] = -1

        self.haar_weights[2, 0, 1, 0] = -1
        self.haar_weights[2, 0, 1, 1] = -1

        self.haar_weights[3, 0, 1, 0] = -1
        self.haar_weights[3, 0, 0, 1] = -1

        self.haar_weights = torch.cat([self.haar_weights] * self.channel_in, 0)
        self.haar_weights = nn.Parameter(self.haar_weights)
        self.haar_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(1/16.)

            out = F.conv2d(x, self.haar_weights, bias=None, stride=2, groups=self.channel_in) / 4.0
            out = out.reshape([x.shape[0], self.channel_in, 4, x.shape[2] // 2, x.shape[3] // 2])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2] // 2, x.shape[3] // 2])
            return out
        else:
            self.elements = x.shape[1] * x.shape[2] * x.shape[3]
            self.last_jac = self.elements / 4 * np.log(16.)

            out = x.reshape([x.shape[0], 4, self.channel_in, x.shape[2], x.shape[3]])
            out = torch.transpose(out, 1, 2)
            out = out.reshape([x.shape[0], self.channel_in * 4, x.shape[2], x.shape[3]])
            return F.conv_transpose2d(out, self.haar_weights, bias=None, stride=2, groups = self.channel_in)

    def jacobian(self, x, rev=False):
        return self.last_jac


class ConvDownsampling(nn.Module):
    def __init__(self, scale):
        super(ConvDownsampling, self).__init__()
        self.scale = scale
        self.scale2 = self.scale ** 2

        self.conv_weights = torch.eye(self.scale2)

        if self.scale == 2: # haar init
            self.conv_weights[0] = torch.Tensor([1./4, 1./4, 1./4, 1./4])
            self.conv_weights[1] = torch.Tensor([1./4, -1./4, 1./4, -1./4])
            self.conv_weights[2] = torch.Tensor([1./4, 1./4, -1./4, -1./4])
            self.conv_weights[3] = torch.Tensor([1./4, -1./4, -1./4, 1./4])
        else:
            self.conv_weights[0] = torch.Tensor([1./(self.scale2)] * (self.scale2))

        self.conv_weights = nn.Parameter(self.conv_weights)

    def forward(self, x, rev=False):
        if not rev:
            # downsample
            # may need improvement
            h = x.shape[2]
            w = x.shape[3]
            wpad = 0
            hpad = 0
            if w % self.scale != 0:
                wpad = self.scale - w % self.scale
            if h % self.scale != 0:
                hpad = self.scale - h % self.scale
            if wpad != 0 or hpad != 0:
                padding = (wpad // 2, wpad - wpad // 2, hpad // 2, hpad - hadp // 2)
                pad = nn.ReplicationPad2d(padding)
                x = pad(x)

            [B, C, H, W] = list(x.size())
            x = x.reshape(B, C, H // self.scale, self.scale, W // self.scale, self.scale)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.reshape(B, C * self.scale2, H // self.scale, W // self.scale)

            # conv
            conv_weights = self.conv_weights.reshape(self.scale2, self.scale2, 1, 1)
            conv_weights = conv_weights.repeat(C, 1, 1, 1)

            out = F.conv2d(x, conv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale2, H // self.scale, W // self.scale)
            out = torch.transpose(out, 1, 2)
            out = out.reshape(B, C * self.scale2, H // self.scale, W // self.scale)

            return out
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.scale2, self.scale2, 1, 1)

            [B, C_, H_, W_] = list(x.size())
            C = C_ // self.scale2
            H = H_ * self.scale
            W = W_ * self.scale

            inv_weights = inv_weights.repeat(C, 1, 1, 1)

            x = x.reshape(B, self.scale2, C, H_, W_)
            x = torch.transpose(x, 1, 2)
            x = x.reshape(B, C_, H_, W_)

            out = F.conv2d(x, inv_weights, bias=None, stride=1, groups=C)

            out = out.reshape(B, C, self.scale, self.scale, H_, W_)
            out = out.permute(0, 1, 4, 2, 5, 3)
            out = out.reshape(B, C, H, W)

            return out


class Conv1x1Grey(nn.Module):
    def __init__(self, rgb_type, learnable=True):
        super(Conv1x1Grey, self).__init__()

        self.channel_in = 3
        self.conv_weights = torch.eye(self.channel_in)
        if rgb_type == 'RGB':
            self.conv_weights[0] = torch.Tensor([0.299, 0.587, 0.114])
            self.conv_weights[1] = torch.Tensor([-0.147, -0.289, 0.436])
            self.conv_weights[2] = torch.Tensor([0.615, -0.515, -0.100])
        elif rgb_type == 'BGR':
            self.conv_weights[0] = torch.Tensor([0.114, 0.587, 0.299])
            self.conv_weights[1] = torch.Tensor([0.436, -0.289, -0.147])
            self.conv_weights[2] = torch.Tensor([-0.100, -0.515, 0.615])
        else:
            print("Error! Undefined RGB type!")
            exit(1)

        self.conv_weights = nn.Parameter(self.conv_weights)

        if not learnable:
            self.conv_weights.requires_grad = False

    def forward(self, x, rev=False):
        if not rev:
            conv_weights = self.conv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            out = F.conv2d(x, conv_weights, bias=None, stride=1)
            return out
        else:
            inv_weights = torch.inverse(self.conv_weights)
            inv_weights = inv_weights.reshape(self.channel_in, self.channel_in, 1, 1)
            out = F.conv2d(x, inv_weights, bias=None, stride=1)
            return out


class InvGreyNet(nn.Module):
    def __init__(self, rgb_type, subnet_constructor=None, block_num=[], Conv1x1Grey_learnable=True):
        super(InvGreyNet, self).__init__()

        channel_in = 3
        channel_out = 1

        operations = []

        b = Conv1x1Grey(rgb_type, Conv1x1Grey_learnable)
        operations.append(b)

        for j in range(block_num[0]):
            b = InvBlockExp(subnet_constructor, channel_in, channel_out)
            operations.append(b)

        self.operations = nn.ModuleList(operations)

    def forward(self, x, rev=False, cal_jacobian=False):
        out = x
        jacobian = 0

        if not rev:
            for op in self.operations:
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)
        else:
            for op in reversed(self.operations):
                out = op.forward(out, rev)
                if cal_jacobian:
                    jacobian += op.jacobian(out, rev)

        if cal_jacobian:
            return out, jacobian
        else:
            return out

