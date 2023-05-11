import numpy as np
import torch
from collections import OrderedDict


def loadmodel(sr_model, sr_pretrain, NGPUs):
    if not sr_pretrain:
        print('\t No pretrained weights provided.')
        return sr_model
    print(f'\t Load pretrained weights from {sr_pretrain}.')
    weights = torch.load(sr_pretrain)
    if 'params_ema' in weights.keys():
        weights = weights['params_ema']
    if 'params' in weights.keys():
        weights = weights['params']
    weights_dict = OrderedDict()
    for k, v in weights.items():
        if k.startswith('module.'):
            weights_dict[k[7:]] = v
        else:
            weights_dict[k] = v
    sr_model.load_state_dict(weights_dict, strict=True)
    sr_model = sr_model.to('cuda')
    sr_model = torch.nn.DataParallel(sr_model, range(NGPUs))
    sr_model.eval()
    return sr_model


def forward_pad(x, forward_function, scale, times=4):
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


def forward_chop(x, forward_function, scale, n_GPUs, multi_output=False, shave=10, min_size=None):
    if not min_size:
        min_size = 160000
    n_GPUs = min(n_GPUs, 4)
    multi_frame = len(x.size()) == 5
    if multi_frame:
        b, f, c, h, w = x.size()
    else:
        b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    lr_list = [
        x[..., 0:h_size, 0:w_size],
        x[..., 0:h_size, (w - w_size):w],
        x[..., (h - h_size):h, 0:w_size],
        x[..., (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        sr_list = []
        for i in range(0, 4, n_GPUs):
            lr_batch = torch.cat(lr_list[i:(i + n_GPUs)], dim=0)
            sr_batch = forward_function(lr_batch)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, forward_function, scale, n_GPUs,
                                shave=shave, min_size=min_size, multi_output=multi_output)
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    if multi_output:
        output = x.new(b, sr_list[0].shape[1], c, h, w)
    else:
        output = x.new(b, c, h, w)
    output[..., 0:h_half, 0:w_half] \
        = sr_list[0][..., 0:h_half, 0:w_half]
    output[..., 0:h_half, w_half:w] \
        = sr_list[1][..., 0:h_half, (w_size - w + w_half):w_size]
    output[..., h_half:h, 0:w_half] \
        = sr_list[2][..., (h_size - h + h_half):h_size, 0:w_half]
    output[..., h_half:h, w_half:w] \
        = sr_list[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

