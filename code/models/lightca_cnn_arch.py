import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from registry import MODEL_REGISTRY

@MODEL_REGISTRY.register()
class LightCACnn(nn.Module):
    def __init__(self, num_feat=64, num_block=3):
        super(LightCACnn, self).__init__()
        self.num_feat = num_feat
        self.num_block = num_block

        # Activation Function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.conv_first = nn.Conv2d(3, num_feat, 3, 1, 1)
        self.downshuffle = PixelShuffleD(0.5)
        self.conv_base = nn.Conv2d(num_feat * 4, num_feat, 3, 1, 1)

        self.conv_1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.blocks = nn.ModuleList()
        for i in range(num_block):
            self.blocks.append(Block(num_feat, self.lrelu))

        self.conv_merge1 = nn.Conv2d(num_feat * (num_block + 1), num_feat, 1, 1, 0)
        self.conv_merge2 = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        self.upshuffle = PixelShuffleD(2)
        self.conv_last = nn.Conv2d(num_feat // 4, 3, 3, 1, 1)


    def _forward(self, x):
        feats = []
        out = self.conv_first(x)
        out = self.downshuffle(out)
        base = self.lrelu(self.conv_base(out))

        out = self.lrelu(self.conv_1(base))
        out = self.conv_2(out)
        feats.append(out)
        for i in range(self.num_block):
            out = torch.cat([out, base], dim=1)
            out = self.blocks[i](out)
            feats.append(out)

        out = self.lrelu(self.conv_merge1(torch.cat(feats, dim=1)))
        out = self.lrelu(self.conv_merge2(torch.cat([out, base], dim=1)))
        out = self.upshuffle(out)
        out = self.lrelu(self.conv_last(out))
        return out

    def forward(self, x):
        return self.forward_pad(x, self._forward, 1, 2)

    def forward_pad(self, x, forward_function, scale, times=4):
        multi_frame = len(x.size()) == 5
        if multi_frame:
            b,n,c,h,w = x.size()
        else:
            b,c,h,w = x.size()
        h_n = int(times*np.ceil(h/times))
        w_n = int(times*np.ceil(w/times))
        if multi_frame:
            imgs_temp = x.new_zeros(b,n,c,h_n,w_n)
        else:
            imgs_temp = x.new_zeros(b,c,h_n,w_n)
        imgs_temp[..., :h, :w] = x
        model_output = forward_function(imgs_temp)
        output = model_output[..., :scale*h, :scale*w]
        return output

class Block(nn.Module):
    def __init__(self, num_feat, act):
        super(Block, self).__init__()
        self.act = act
        self.conv_merge = nn.Conv2d(num_feat * 2, num_feat, 1, 1, 0)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        x = self.act(self.conv_merge(x))
        x = self.act(self.conv1(x))
        x = self.conv2(x)
        return x

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
