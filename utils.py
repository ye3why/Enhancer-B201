import numpy as np
import cv2
import ffmpeg
import torch
import torchvision
from registry import SAVEIMG_REGISTRY

def flatten_list(li):
    return sum(([x] if not isinstance(x, list) and not isinstance(x, tuple) else flatten_list(x) for x in li), [])

@SAVEIMG_REGISTRY.register('sdr')
def saveimg_sdr(img, path):
    return torchvision.utils.save_image(img, path)

def colortrans_709_2020(img, gamma):
    img = img**gamma
    h,w,c = img.shape
    m1 = np.array([[0.6274, 0.3293, 0.0433],
       [0.0691, 0.9195, 0.0114],
       [0.0164, 0.0880, 0.8956]])
    imgnew = img.transpose((2, 0, 1))
    imgnew = imgnew.reshape((c,-1))
    imgnew = np.dot(m1,imgnew)
    imgnew = np.clip(imgnew, 0, 1)
    imgnew = imgnew.reshape((c,h,w))
    img = imgnew.transpose((1, 2, 0))
    img = img**(1/gamma)
    return img

@SAVEIMG_REGISTRY.register('hlg')
def saveimg_hlg(img, path):

    img = img.cpu().detach().numpy().transpose((1, 2, 0))
    img = img / 1023.0
    img[img>1]=1
    img[img<0]=0
    img = colortrans_709_2020(img, 2.4)
    img = (img * 65535.0).astype(np.uint16)
    img = img[:,:,::-1].copy()
    cv2.imwrite(str(path), img)

def save_image(imgtensor, filename):
    _tensor = imgtensor.float().detach().cpu().clamp_(0, 1) * 65535
    img_np = _tensor.numpy().astype('uint16')
    img_np = np.transpose(img_np[[2, 1, 0], :, :], (1, 2, 0))  # HWC, BGR
    cv2.imwrite(str(filename), img_np)


def get_nb_frames(videopath):
    info = ffmpeg.probe(videopath)
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            return stream['nb_frames']

def get_codec(video):
    info = ffmpeg.probe(str(video))
    for s in info['streams']:
        if s['codec_type'] == 'video':
            return s['codec_name']

def get_ffmpeg_args(video):
    info = ffmpeg.probe(str(video))
    res = {}
    res['has_audio'] = False
    for s in info['streams']:
        if s['codec_type'] == 'video':
            res['r'] = s['r_frame_rate']
            res['pix_fmt'] = s['pix_fmt']
            res['b:v'] = s.get('bit_rate', info['format']['bit_rate'])
            res['height'] = s['height']
            res['width'] = s['width']
        if s['codec_type'] == 'audio':
            res['has_audio'] = True
    return res

def input_cycle(x, target_n_frames):
    res = []
    x_nframes = x.shape[1]
    center_idx = x_nframes // 2
    half_n = target_n_frames // 2
    for i in range(x_nframes):
        tmp = x
        tmp[:, center_idx], tmp[:, i] = tmp[:, i], tmp[:, center_idx]
        res.append(tmp)
    res = torch.cat(res, dim=0)
    return res[:, center_idx - half_n: center_idx + half_n + 1]

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
        output = x.new(b, f, c, h, w)
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

