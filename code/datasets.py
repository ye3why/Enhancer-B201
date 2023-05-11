import torch
import torch.utils.data as data
import torchvision
import cv2
import scipy.misc as misc
from pathlib import Path

from registry import DATASET_REGISTRY
import utils

@DATASET_REGISTRY.register()
class ImageDataset(data.Dataset):
    def __init__(self, video_dir, n_frames):
        super(ImageDataset, self).__init__()
        # self.img_path = list(Path(video_dir).glob('*.png'))
        self.img_path = utils.glob_pic(video_dir)
        self.video_len = len(self.img_path)
        self.transformer = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        lr_path = self.img_path[idx]
        filename = lr_path.name
        img_path = lr_path
        # frame_idx = int(lr_path.stem)
        # img_path = lr_path.parent.joinpath('{}.png'.format(frame_idx))
        # lr = misc.imread(img_path)
        lr = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
        lr = lr[:, :, [2, 1, 0]]
        lr = self.transformer(lr) # [0.0, 1.0]
        return lr, filename

    def __len__(self):
        return len(self.img_path)


@DATASET_REGISTRY.register()
class VideoDataset(data.Dataset):
    def __init__(self, video_dir, n_frames):
        super(VideoDataset, self).__init__()
        self.n_frames = n_frames
        # self.img_path = list(Path(video_dir).glob('*.png'))
        self.img_path = sorted(utils.glob_pic(video_dir))
        self.video_len = len(self.img_path)
        # self.video_idx_lower_bound = min([int(img.stem) for img in self.img_path])
        # self.video_idx_upper_bound = max([int(img.stem) for img in self.img_path])
        self.transformer = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        lr_path = self.img_path[idx]
        filename = lr_path.name
        # frame_idx = int(lr_path.stem)
        # idxs = self.index_generation(
            # frame_idx, self.video_idx_upper_bound, self.n_frames, self.video_idx_lower_bound)
        idxs = self.index_generation(idx, self.video_len - 1, self.n_frames, 0)
        lrs = []
        for neighbor_idx in idxs:
            img_path = self.img_path[neighbor_idx]
            # img_path = lr_path.parent.joinpath('{}.png'.format(idx))
            # temp = misc.imread(img_path)
            temp = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            temp = temp[:, :, [2, 1, 0]]
            temp = self.transformer(temp) # [0.0, 1.0]
            lrs.append(temp)
        lrs = torch.stack(lrs, dim=0)

        return lrs, filename

    def __len__(self):
        return len(self.img_path)

    def index_generation(self, crt_i, upper_bound, N, lower_bound=1):
        n_pad = N // 2
        return_l = []

        for i in range(crt_i - n_pad, crt_i + n_pad + 1):
            if i < lower_bound:
                # add_idx = crt_i + lower_bound - i + n_pad
                add_idx = lower_bound
            elif i > upper_bound:
                # add_idx = crt_i - i + upper_bound - n_pad
                add_idx = upper_bound
            else:
                add_idx = i
            return_l.append(add_idx)
        return return_l

@DATASET_REGISTRY.register()
class FrameInterpDataset(VideoDataset):
    '''Video Dataset for Video frame interplation
    please set model configuration in models.yml, multi_output: true
    '''
    def __init__(self, video_dir, n_frames):
        super(FrameInterpDataset, self).__init__(video_dir, n_frames)

    def __getitem__(self, idx):
        lr_path = self.img_path[idx]
        stem, suffix = lr_path.stem, lr_path.suffix
        # filename = [str(int(stem)*2-1) + suffix, str(int(stem)*2) + suffix]
        filename = ['{:>08d}'.format(int(stem)*2-1) + suffix, '{:>08d}'.format(int(stem)*2) + suffix]
        idxs = self.index_generation(idx, self.video_len - 1, self.n_frames, 0)
        lrs = []
        for neighbor_idx in idxs:
            img_path = self.img_path[neighbor_idx]
            # temp = misc.imread(img_path)
            temp = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            temp = temp[:, :, [2, 1, 0]]
            temp = self.transformer(temp) # [0.0, 1.0]
            lrs.append(temp)
        lrs = torch.stack(lrs, dim=0)

        return lrs, filename


@DATASET_REGISTRY.register()
class VideoMinMoutDataset(data.Dataset):
    '''Video Dataset for multi-frame input multi-frame output models
    please set model configuration in models.yml, multi_output: true
    '''
    def __init__(self, video_dir, n_frames):
        super(VideoMinMoutDataset, self).__init__()
        self.n_frames = n_frames
        # self.img_path = list(Path(video_dir).glob('*.png'))
        self.img_path = sorted(utils.glob_pic(video_dir))
        self.clip_idx = list(range(0, len(self.img_path), n_frames))
        self.transformer = torchvision.transforms.ToTensor()

    def __getitem__(self, idx):
        lr_paths = self.img_path[self.clip_idx[idx]:min(self.clip_idx[idx]+self.n_frames, len(self.img_path))]
        filenames = [path.name for path in lr_paths]
        lrs = []
        for img_path in lr_paths:
            # temp = misc.imread(img_path)
            temp = cv2.imread(str(img_path), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
            temp = temp[:, :, [2, 1, 0]]
            temp = self.transformer(temp) # [0.0, 1.0]
            lrs.append(temp)
        lrs = torch.stack(lrs, dim=0)

        return lrs, filenames

    def __len__(self):
        return len(self.clip_idx)
