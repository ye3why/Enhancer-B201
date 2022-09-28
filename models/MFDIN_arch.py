''' network architecture for MFDIN pytorch1.7'''
import functools
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
import torchvision.ops as OP
from registry import MODEL_REGISTRY


def initialize_weights(net_l, scale=1):
    if not isinstance(net_l, list):
        net_l = [net_l]
    for net in net_l:
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale  # for residual block
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias.data, 0.0)


def make_layer(block, n_layers):
    layers = []
    for _ in range(n_layers):
        layers.append(block())
    return nn.Sequential(*layers)

class BasicConv(nn.Module):
    def __init__(self, nf=64):
        super(BasicConv, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)  #0.1的比例初始化权重

    def forward(self, x):
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return out

class RFABlock(nn.Module):
    def __init__(self, nf=64):
        super(RFABlock, self).__init__()
        self.res1 = BasicConv(nf)
        self.res2 = BasicConv(nf)
        self.res3 = BasicConv(nf)
        self.res4 = BasicConv(nf)
        self.conv = nn.Conv2d(nf*4, nf, 1, bias=False)

    def forward(self, x):
        identity = x
        fea1 = self.res1(x)
        xin2 = identity+fea1
        fea2 = self.res2(xin2)
        xin3 = xin2+fea2
        fea3 = self.res3(xin3)
        xin4 = xin3+fea3
        fea4 = self.res4(xin4)
        out =  self.conv(torch.cat([fea1,fea2,fea3,fea4],dim=1))

        return identity + out

class ResidualBlock_noBN(nn.Module):
    '''Residual block w/o BN
    ---Conv-ReLU-Conv-+-
     |________________|
    '''

    def __init__(self, nf=64):
        super(ResidualBlock_noBN, self).__init__()
        self.conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)

        # initialization
        initialize_weights([self.conv1, self.conv2], 0.1)  #0.1的比例初始化权重

    def forward(self, x):
        identity = x
        out = F.relu(self.conv1(x), inplace=True)
        out = self.conv2(out)
        return identity + out


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros'):
    """Warp an image or feature map with optical flow
    Args:
        x (Tensor): size (N, C, H, W)
        flow (Tensor): size (N, H, W, 2), normal value
        interp_mode (str): 'nearest' or 'bilinear'
        padding_mode (str): 'zeros' or 'border' or 'reflection'

    Returns:
        Tensor: warped image or feature map
    """
    assert x.size()[-2:] == flow.size()[1:3]
    B, C, H, W = x.size()
    # mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False
    grid = grid.type_as(x)
    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(W - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(H - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode)
    return output

class Conv_Blocks(nn.Module):
    def __init__(self, in_channel, out_channel, downsamle=False, upsample=False):
        super(Conv_Blocks, self).__init__()
        self.upsample = upsample
        if not downsamle:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )
        else:
            self.layer = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, 2, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True),
                nn.Conv2d(out_channel, out_channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=True)
            )

    def forward(self, x):
        x = self.layer(x)
        if self.upsample:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = x*2
        return x

class Encoder_Decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Encoder_Decoder, self).__init__()
        self.in_conv = Conv_Blocks(in_channel, 64, downsamle=False, upsample=False)
        self.down1 = Conv_Blocks(64, 64, downsamle=True, upsample=False)
        self.down2 = Conv_Blocks(64, 64, downsamle=True, upsample=False)
        self.down_up = Conv_Blocks(64, 64, downsamle=True, upsample=True)
        self.up1 = Conv_Blocks(64, 64, downsamle=False, upsample=True)
        self.up2 = Conv_Blocks(64, 64, downsamle=False, upsample=True)
        self.out_conv = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.out_mask_offset = nn.Sequential(
            nn.Conv2d(64, out_channel, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(out_channel, out_channel, 3, 1, 1)
        )

    def forward(self, x):
        f1 = self.in_conv(x)
        f2 = self.down1(f1)  #64
        f3 = self.down2(f2)  #32
        f4 = self.down_up(f3) #32
        f5 = self.up1(f3 + f4) #64
        f6 = self.up2(f5 + f2)
        f = self.out_conv(f6 + f1)
        return self.out_mask_offset(f)

MotionNet = Encoder_Decoder

class D_Align(nn.Module):
    def __init__(self, nf=64, groups=4):
        super(D_Align, self).__init__()
        self.nf = nf
        self.cas_offset_conv1 = nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=True)
        self.offsetnet = MotionNet(nf, 2 * groups * 3 * 3)
        self.cas_dcnpack = OP.DeformConv2d(nf, nf, 3, padding=1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, nbr_fea_l, ref_fea_l):
        offset = self.lrelu(self.cas_offset_conv1(torch.cat([nbr_fea_l, ref_fea_l], dim=1)))
        offset = self.offsetnet(offset)
        L1_fea = self.lrelu(self.cas_dcnpack(nbr_fea_l, offset))  # 对L1对齐帧形变对齐
        return L1_fea  # 最终对齐帧

@MODEL_REGISTRY.register()
class MFDIN_2X2P(nn.Module):
    def __init__(self, nf=64, groups=4, front_RBs=5, back_RFAs=2, center=None, nfields=5):
        super(MFDIN_2X2P, self).__init__()
        self.nf = nf
        self.center = nfields // 2 if center is None else center
        ResidualBlock_noBN_f = functools.partial(ResidualBlock_noBN, nf=nf)
        RFABlock_ = functools.partial(RFABlock, nf=nf)
        #### extract features (for each frame)
        self.conv_first = nn.Conv2d(3, nf, 3, 1, 1, bias=True)
        self.feature_extraction = make_layer(ResidualBlock_noBN_f, front_RBs)  # 前残差组提取特征||test分组卷积
        self.fea_upconv = nn.Conv2d(nf, nf * 2, 3, 1, 1, bias=True)
        self.fea_conv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.AlignMoudle = D_Align(nf=nf, groups=groups)
        self.fusion = nn.Conv2d(nfields * nf, nf, 1, 1, bias=True)  # 普通融合卷积函数
        #### reconstruction
        self.upconv1 = nn.Conv2d(nf, nf * 4, 3, 1, 1, bias=True)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.recon_trunk = make_layer(RFABlock_, back_RFAs)  # 后残差组重建图片
        self.HRconv = nn.Conv2d(64, 64, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(64, 3, 3, 1, 1, bias=True)
        #### activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
    def V_shuffle(self, inputs, scale):
        N, C, iH, iW = inputs.size()
        oH = iH * scale
        oW = iW
        oC = C // scale
        output = inputs.view(N, oC, scale, iH, iW)
        output = output.permute(0, 1, 3, 2, 4).contiguous()
        output = output.view(N, oC, oH, oW)
        return output

    def forward(self, x):
        B, N, C, H, W = x.size()  # N video frames
        fields = []
        fields.append(x[:, 0, :, :, :][:, :, ::2, :])
        fields.append(x[:, 0, :, :, :][:, :, 1::2, :])
        fields.append(x[:, 1, :, :, :][:, :, ::2, :])
        fields.append(x[:, 1, :, :, :][:, :, 1::2, :])
        fields.append(x[:, 2, :, :, :][:, :, ::2, :])
        fields.append(x[:, 2, :, :, :][:, :, 1::2, :])
        fields = torch.stack(fields, dim=1)
        N = 6
        #### extract LR features
        fea1 = self.lrelu(self.conv_first(fields.view(-1, C, H // 2, W)))
        f_fea = self.feature_extraction(fea1)  # 前残差组：提取特征
        f_fea = self.lrelu(self.fea_upconv(f_fea + fea1))
        f_fea = self.lrelu(self.fea_conv1(self.V_shuffle(f_fea, 2)))
        f_fea = f_fea.view(B, N, -1, H, W)  # [4, 6, 64, 64, 64]

        #### pcd align
        ref_fea_lA = f_fea[:, self.center, :, :, :].clone()  # 设定每个层的参考帧
        ref_fea_lB = f_fea[:, self.center + 1, :, :, :].clone()
        aligned_feaA = []
        aligned_feaB = []
        for i in range(N - 1):
            nbr_fea_lA = f_fea[:, i, :, :, :].clone()
            nbr_fea_lB = f_fea[:, i + 1, :, :, :].clone()
            aligned_feaA.append(self.AlignMoudle(nbr_fea_lA, ref_fea_lA))
            aligned_feaB.append(self.AlignMoudle(nbr_fea_lB, ref_fea_lB))
        aligned_feaA = torch.stack(aligned_feaA, dim=1).view(B, -1, H, W)  # [B, N, C, H, W]
        aligned_feaB = torch.stack(aligned_feaB, dim=1).view(B, -1, H, W)
        feaA = self.fusion(aligned_feaA)  # TSA融合模块
        feaB = self.fusion(aligned_feaB)
        fea = torch.stack((feaA, feaB), dim=1).view(B * 2, -1, H, W)
        out = self.recon_trunk(fea)  # 重建模块
        out = self.lrelu(self.pixel_shuffle(self.upconv1(out)))
        out = self.lrelu(self.HRconv(out))
        out = self.conv_last(out).view(B,2,3,H*2,W*2)

        return out

@MODEL_REGISTRY.register()
class MFDIN_2X(MFDIN_2X2P):
    def __init__(self, nf=64, groups=4, front_RBs=5, back_RFAs=2, center=None, nfields=5):
        super(MFDIN_2X, self).__init__(nf, groups, front_RBs, back_RFAs, center, nfields)

    def forward(self, x):
        out = super().forward(x)
        out = out[:, 0]
        return out

@MODEL_REGISTRY.register()
class MFDIN_2P(MFDIN_2X2P):
    def __init__(self, nf=64, groups=4, front_RBs=5, back_RFAs=2, center=None, nfields=5):
        super(MFDIN_2P, self).__init__(nf, groups, front_RBs, back_RFAs, center, nfields)

    def forward(self, x):
        out = super().forward(x)
        out = F.interpolate(out, scale_factor=1/2)
        return out

@MODEL_REGISTRY.register()
class MFDIN_DeInterlace(MFDIN_2X2P):
    def __init__(self, nf=64, groups=4, front_RBs=5, back_RFAs=2, center=None, nfields=5):
        super(MFDIN_DeInterlace, self).__init__(nf, groups, front_RBs, back_RFAs, center, nfields)

    def forward(self, x):
        out = super().forward(x)
        out = out[:, 0]
        out = F.interpolate(out, scale_factor=1/2)
        return out






