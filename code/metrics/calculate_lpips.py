import cv2
import argparse
import glob
import torch
import numpy as np
import os.path as osp
from torchvision.transforms.functional import normalize

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')

from .metric_util import *
from registry import METRIC_REGISTRY

@METRIC_REGISTRY.register('lpips')
class LPIPSWorker(EvalWorker):
    def __init__(self, que, qid, **kwargs):
        super(LPIPSWorker, self).__init__(que, qid)
        # self.__dict__.update(kwargs)
        self.loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]

    @torch.no_grad()
    def eval_func(self, res, name, restored, gt=None):
        img_gt = gt / 255
        img_restored = restored / 255
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = self.loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).flatten()[0]
        res[name] = lpips_val

    @staticmethod
    def need_gt():
        return True

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=str, default='datasets/val_set14/Set14', help='Path to gt (Ground-Truth)')
    parser.add_argument('--restored', type=str, default='results/Set14', help='Path to restored images')
    args = parser.parse_args()
    return args

def main(folder_gt, folder_restored):
    # Configurations
    # -------------------------------------------------------------------------
    # folder_gt = 'datasets/celeba/celeba_512_validation'
    # folder_restored = 'datasets/celeba/celeba_512_validation_lq'
    # crop_border = 4
    suffix = ''
    # -------------------------------------------------------------------------
    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    lpips_all = []
    img_list = sorted(glob.glob(osp.join(folder_gt, '*')))

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    for i, img_path in enumerate(img_list):
        basename, ext = osp.splitext(osp.basename(img_path))
        img_gt = cv2.imread(img_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.
        img_restored = cv2.imread(osp.join(folder_restored, basename + suffix + ext), cv2.IMREAD_UNCHANGED).astype(
            np.float32) / 255.

        img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
        # norm to [-1, 1]
        normalize(img_gt, mean, std, inplace=True)
        normalize(img_restored, mean, std, inplace=True)

        # calculate lpips
        lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).flatten()[0]

        print(f'{i+1:3d}: {basename:25}. \tLPIPS: {lpips_val:.6f}.')
        lpips_all.append(lpips_val)

    print(folder_gt)
    print(folder_restored)
    print(f'Average: LPIPS: {sum(lpips_all) / len(lpips_all):.6f}')


@torch.no_grad()
def calculate_lpips(img_gt, img_restored):
    img_gt = img_gt / 255
    img_restored = img_restored / 255
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    img_gt, img_restored = img2tensor([img_gt, img_restored], bgr2rgb=True, float32=True)
    # norm to [-1, 1]
    normalize(img_gt, mean, std, inplace=True)
    normalize(img_restored, mean, std, inplace=True)

    loss_fn_vgg = lpips.LPIPS(net='vgg').cuda()  # RGB, normalized to [-1,1]
    # calculate lpips
    lpips_val = loss_fn_vgg(img_restored.unsqueeze(0).cuda(), img_gt.unsqueeze(0).cuda()).flatten()[0]

    return lpips_val


if __name__ == '__main__':
    args = parse_args()
    with torch.no_grad():
        main(args.gt, args.restored)
