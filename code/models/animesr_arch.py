import random
import torch
from torch import nn as nn
from torch.nn import functional as F
from registry import MODEL_REGISTRY
from .srvgg_arch import SRVGGNetCompact
from .rrdbnet_arch import RRDBNet


@MODEL_REGISTRY.register()
class AnimeSR(SRVGGNetCompact):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu'):
        super(AnimeSR, self).__init__(num_in_ch, num_out_ch, num_feat, num_conv, 4, act_type)
        self.outscale= upscale

    def forward(self, x):
        out = super().forward(x)
        return F.interpolate(out, scale_factor=self.outscale/4, mode='bicubic')



@MODEL_REGISTRY.register()
class AnimeSRRRDB(RRDBNet):
    def __init__(self, num_in_ch, num_out_ch, upscale=4, num_feat=64, num_block=23, num_grow_ch=32):
        super(AnimeSRRRDB, self).__init__(num_in_ch, num_out_ch, 4, num_feat, num_block, num_grow_ch)
        self.outscale= upscale

    def forward(self, x):
        out = super().forward(x)
        return F.interpolate(out, scale_factor=self.outscale/4, mode='bicubic')

def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)
class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class RightAlignMSConvResidualBlocks(nn.Module):
    """right align multi-scale ConvResidualBlocks, currently only support 3 scales (1, 2, 4)"""

    def __init__(self, num_in_ch=3, num_state_ch=64, num_out_ch=64, num_block=(5, 3, 2)):
        super().__init__()

        assert len(num_block) == 3
        assert num_block[0] >= num_block[1] >= num_block[2]
        self.num_block = num_block

        self.conv_s1_first = nn.Sequential(
            nn.Conv2d(num_in_ch, num_state_ch, 3, 1, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s2_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True), nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.conv_s4_first = nn.Sequential(
            nn.Conv2d(num_state_ch, num_state_ch, 3, 2, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
        )

        self.body_s1_first = nn.ModuleList()
        for _ in range(num_block[0]):
            self.body_s1_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s2_first = nn.ModuleList()
        for _ in range(num_block[1]):
            self.body_s2_first.append(ResidualBlockNoBN(num_feat=num_state_ch))
        self.body_s4_first = nn.ModuleList()
        for _ in range(num_block[2]):
            self.body_s4_first.append(ResidualBlockNoBN(num_feat=num_state_ch))

        self.upsample_x2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.upsample_x4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False)

        self.fusion = nn.Sequential(
            nn.Conv2d(3 * num_state_ch, 2 * num_out_ch, 3, 1, 1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2 * num_out_ch, num_out_ch, 3, 1, 1, bias=True),
        )

    def up(self, x, scale=2):
        if isinstance(x, int):
            return x
        elif scale == 2:
            return self.upsample_x2(x)
        else:
            return self.upsample_x4(x)

    def forward(self, x):
        x_s1 = self.conv_s1_first(x)
        x_s2 = self.conv_s2_first(x_s1)
        x_s4 = self.conv_s4_first(x_s2)

        flag_s2 = False
        flag_s4 = False
        for i in range(0, self.num_block[0]):
            x_s1 = self.body_s1_first[i](
                x_s1 + (self.up(x_s2, 2) if flag_s2 else 0) + (self.up(x_s4, 4) if flag_s4 else 0))
            if i >= self.num_block[0] - self.num_block[1]:
                x_s2 = self.body_s2_first[i - self.num_block[0] + self.num_block[1]](
                    x_s2 + (self.up(x_s4, 2) if flag_s4 else 0))
                flag_s2 = True
            if i >= self.num_block[0] - self.num_block[2]:
                x_s4 = self.body_s4_first[i - self.num_block[0] + self.num_block[2]](x_s4)
                flag_s4 = True

        x_fusion = self.fusion(torch.cat((x_s1, self.upsample_x2(x_s2), self.upsample_x4(x_s4)), dim=1))

        return x_fusion

@MODEL_REGISTRY.register()
class MSRSWVSR(nn.Module):
    """
    Multi-Scale, unidirectional Recurrent, Sliding Window (MSRSW)
    The implementation refers to paper: Efficient Video Super-Resolution through Recurrent Latent Space Propagation
    """

    def __init__(self, num_feat=64, num_block=(5, 3, 2), netscale=4):
        super(MSRSWVSR, self).__init__()
        self.num_feat = num_feat

        # 3(img channel) * 3(prev cur nxt 3 imgs) + 3(hr img channel) * netscale * netscale + num_feat
        self.recurrent_cell = RightAlignMSConvResidualBlocks(3 * 3 + 3 * netscale * netscale + num_feat, num_feat,
                                                             num_feat + 3 * netscale * netscale, num_block)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1)
        self.pixel_shuffle = nn.PixelShuffle(netscale)
        self.netscale = netscale
        self.dirname = None
        self.state, self.out = None, None

    def cell(self, x, fb, state):
        res = x[:, 3:6]
        # pre frame, cur frame, nxt frame, pre sr frame, pre hidden state
        inp = torch.cat((x, pixel_unshuffle(fb, self.netscale), state), dim=1)
        # the out contains both state and sr frame
        out = self.recurrent_cell(inp)
        out_img = self.pixel_shuffle(out[:, :3 * self.netscale * self.netscale]) + F.interpolate(
            res, scale_factor=self.netscale, mode='bilinear', align_corners=False)
        out_state = self.lrelu(out[:, 3 * self.netscale * self.netscale:])

        return out_img, out_state

    # def forward(self, x):
        # b, n, c, h, w = x.size()
        # # initialize previous sr frame and previous hidden state as zero tensor
        # out = x.new_zeros(b, c, h * self.netscale, w * self.netscale)
        # state = x.new_zeros(b, self.num_feat, h, w)
        # out_l = []
        # for i in range(n):
            # if i == 0:
                # # there is no previous frame for the 1st frame, so reuse 1st frame as previous
                # out, state = self.cell(torch.cat((x[:, i], x[:, i], x[:, i + 1]), dim=1), out, state)
            # elif i == n - 1:
                # # there is no next frame for the last frame, so reuse last frame as next
                # out, state = self.cell(torch.cat((x[:, i - 1], x[:, i], x[:, i]), dim=1), out, state)
            # else:
                # out, state = self.cell(torch.cat((x[:, i - 1], x[:, i], x[:, i + 1]), dim=1), out, state)
            # out_l.append(out.cpu())

        # return torch.stack(out_l, dim=1)


    def forward(self, x, **kwargs):
        b, n, c, h, w = x.size()
        dirname =  kwargs.get('dirname', None)
        if dirname != self.dirname or self.state is None:
            print(f' Got new clip: {dirname}, Refresh model state.')
            self.out = x.new_zeros(b, c, h * self.netscale, w * self.netscale)
            self.state = x.new_zeros(b, self.num_feat, h, w)
            self.dirname = dirname

        prev, cur, nex = x[:, 0], x[:, 1], x[:, 2]
        # initialize previous sr frame and previous hidden state as zero tensor
        self.out, self.state = self.cell(torch.cat((prev, cur, nex), dim=1), self.out, self.state)
        return self.out.clone()
