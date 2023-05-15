import numpy as np
import torch
import math
from collections import OrderedDict


def loadmodel(sr_model, sr_pretrain, NGPUs):
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


def forward_pad(x, forward_function, scale, times=4, **kwargs):
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
    model_output = forward_function(imgs_temp,  **kwargs)
    output = model_output[..., :scale*h, :scale*w]
    return output


def forward_chop(x, forward_function, scale, n_GPUs, shave=10, min_size=None, **kwargs):
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
            sr_batch = forward_function(lr_batch, **kwargs)
            sr_list.extend(sr_batch.chunk(n_GPUs, dim=0))
    else:
        sr_list = [
            forward_chop(patch, forward_function, scale, n_GPUs,
                                shave=shave, min_size=min_size, **kwargs)
            for patch in lr_list
        ]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale
    multi_output = ( len(sr_list[0].shape) == 5 )
    if multi_output:
        output = x.new(b, sr_list[0].shape[1], sr_list[0].shape[2], h, w)
    else:
        output = x.new(b, sr_list[0].shape[1], h, w)
    output[..., 0:h_half, 0:w_half] \
        = sr_list[0][..., 0:h_half, 0:w_half]
    output[..., 0:h_half, w_half:w] \
        = sr_list[1][..., 0:h_half, (w_size - w + w_half):w_size]
    output[..., h_half:h, 0:w_half] \
        = sr_list[2][..., (h_size - h + h_half):h_size, 0:w_half]
    output[..., h_half:h, w_half:w] \
        = sr_list[3][..., (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def forward_tile(img, forward_function, scale, tile_size, tile_pad,  **kwargs):
    '''
        patch size = tile_size + 2 * tile_pad
    '''
    height, width = img.shape[-2:]
    output_height = height * scale
    output_width = width * scale
    output_flag = True

    # start with black image
    tiles_x = math.ceil(width / tile_size)
    tiles_y = math.ceil(height / tile_size)

    # loop over all tiles
    for y in range(tiles_y):
        for x in range(tiles_x):
            # extract tile from input image
            ofs_x = x * tile_size
            ofs_y = y * tile_size
            # input tile area on total image
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # input tile area on total image with padding
            input_start_x_pad = max(input_start_x - tile_pad, 0)
            input_end_x_pad = min(input_end_x + tile_pad, width)
            input_start_y_pad = max(input_start_y - tile_pad, 0)
            input_end_y_pad = min(input_end_y + tile_pad, height)

            # input tile dimensions
            input_tile_width = input_end_x - input_start_x
            input_tile_height = input_end_y - input_start_y
            tile_idx = y * tiles_x + x + 1
            input_tile = img[..., input_start_y_pad:input_end_y_pad, input_start_x_pad:input_end_x_pad]

            # upscale tile
            output_tile = forward_function(input_tile, **kwargs)

            # output tile area on total image
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # output tile area without padding
            output_start_x_tile = (input_start_x - input_start_x_pad) * scale
            output_end_x_tile = output_start_x_tile + input_tile_width * scale
            output_start_y_tile = (input_start_y - input_start_y_pad) * scale
            output_end_y_tile = output_start_y_tile + input_tile_height * scale

            multi_output = ( len(output_tile.shape) == 5 )
            if output_flag:
                output_flag = False
                if multi_output:
                    output_shape = (*output_tile.shape[:3], output_height, output_width)
                else:
                    output_shape = (*output_tile.shape[:2], output_height, output_width)
                output = img.new_zeros(output_shape)

            # put tile into output image
            output[..., output_start_y:output_end_y,
                        output_start_x:output_end_x] = output_tile[..., output_start_y_tile:output_end_y_tile,
                                                                    output_start_x_tile:output_end_x_tile]
    return output
