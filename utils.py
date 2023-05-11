import numpy as np
import cv2
import ffmpeg
import torch
import torchvision
import string
import shutil
import random
import threading
import time
import tempfile
import os
from pathlib import Path
from registry import SAVEIMG_REGISTRY

def flatten_list(li):
    return sum(([x] if not isinstance(x, list) and not isinstance(x, tuple) else flatten_list(x) for x in li), [])

def ifnot_mkdir(path):
    if isinstance(path, Path):
        if not path.exists():
            path.mkdir()
    elif isinstance(path, str):
        if not os.path.exists(path):
            os.mkdir(path)
    else:
        raise NotImplementedError

def isImg(path):
    return path.suffix in ['.jpg', '.bmp', '.png', '.tiff']


@SAVEIMG_REGISTRY.register('sdr')
def saveimg_sdr(img, path):
    # torchvision.utils.save_image is very slow
    # return torchvision.utils.save_image(img, path)
    img = img.cpu().detach().clamp_(0, 1).numpy().transpose((1, 2, 0)) * 255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img)

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



class ImgSaver(threading.Thread):

    def __init__(self, save_func, que, qid):
        super().__init__()
        self._queue = que
        self.qid = qid
        self.save_func = save_func

    def run(self):
        while True:
            msg = self._queue.get()
            if isinstance(msg, str) and msg == 'quit':
                break
            self.save_func(msg['img'], msg['path'])
        # print(f'{self.qid} quit.')


def getTempdir(tempdir_type, opt=None):
    if tempdir_type == 'mem':
        return MemoryTempDir()
    elif tempdir_type == 'disk':
        return DiskTempDir(opt)
    else:
        raise NotImplementedError

class MemoryTempDir():
    def __init__(self):
        self.handler = tempfile.TemporaryDirectory()
        self.Path = Path(self.handler.name)
        self.string = self.handler.name

    def __del__(self):
        self.handler.cleanup()

    def __str__(self):
        return self.string

    def getPath(self):
        return self.Path

    def getstring(self):
        return self.string



class DiskTempDir():
    def __init__(self, opt):
        randomname = ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase + string.digits, 10))
        self.handler = opt['output_dir'].joinpath('_'.join(['tmp', randomname]))
        assert not self.handler.exists(), f'tempdir path: {self.handler.name} exists!'
        self.handler.mkdir()
        self.Path = self.handler
        self.string = str(self.handler)

    def __del__(self):
        shutil.rmtree(self.handler)

    def __str__(self):
        return self.string

    def getPath(self):
        return self.Path

    def getstring(self):
        return self.string
