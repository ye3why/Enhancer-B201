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
