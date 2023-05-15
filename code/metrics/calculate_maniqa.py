import argparse
import cv2
import random
import numpy as np
from os import path as osp
import torch
from pathlib import Path

from .metric_util import *
from registry import METRIC_REGISTRY
from .model_attentionIQA2 import AttentionIQA

MANIQA_MODEL_PATH = Path(__file__).parent.joinpath('../../weights/maniqa_weights_new.pth')

@METRIC_REGISTRY.register('maniqa')
class MANIQAWorker(EvalWorker):
    def __init__(self, que, qid, **kwargs):
        super(MANIQAWorker, self).__init__(que, qid)
        # self.__dict__.update(kwargs)
        assert MANIQA_MODEL_PATH.exists(), \
            f'MANIQA weights: {MANIQA_MODEL_PATH} doesn\'t exist.'
        model = AttentionIQA(
                    embed_dim=768,
                    num_outputs=1,
                    patch_size=8,
                    drop=0.1,
                    depths=[2, 2],
                    window_size=4,
                    dim_mlp=768,
                    num_heads=[4, 4],
                    img_size=224,
                    num_channel_attn=2,
                    num_tab=2,
                    scale=0.13
        )
        weights =torch.load(MANIQA_MODEL_PATH)
        model.load_state_dict(weights, strict=True)
        # map to cuda, if available
        self.cuda_flag = False
        if torch.cuda.is_available():
            model = model.cuda()
            self.cuda_flag = True
        model.eval()
        self.model = model
        self.average_iters = 20
        self.crop_size = 224

    @torch.no_grad()
    def eval_func(self, res, name, restored, gt=None):
        restored = cv2.cvtColor(restored, cv2.COLOR_BGR2RGB)
        restored = np.array(restored).astype('float32') / 255
        restored = np.transpose(restored, (2, 0, 1))
        restored = (restored - 0.5) / 0.5
        restored = torch.from_numpy(restored).type(torch.FloatTensor)
        restored = restored.unsqueeze(0)
        b, c, h, w = restored.shape
        if self.cuda_flag:
            restored = restored.cuda()
        pred = 0
        for i in range(self.average_iters):
            top = random.randint(0, h - self.crop_size)
            left = random.randint(0, w - self.crop_size)
            img = restored[:, :, top:top + self.crop_size, left:left + self.crop_size]
            pred += self.model(img)
        pred /= self.average_iters
        res[name] = pred.cpu().numpy()[0][0]

    @staticmethod
    def need_gt():
        return
