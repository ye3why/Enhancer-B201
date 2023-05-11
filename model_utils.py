import numpy as np
import torch
import copy
import functools
import queue
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import datasets
import utils
import models as models_module
from registry import MODEL_REGISTRY
from registry import DATASET_REGISTRY
from registry import SAVEIMG_REGISTRY

def loadmodel(sr_model, sr_pretrain, NGPUs):
    if not sr_pretrain:
        print('No pretrained weights provided.')
        return sr_model
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

def prepare_model(opt, models_conf):
    prepared_models = {}
    for model_name in opt['models']:
        m = copy.deepcopy(models_conf[model_name])

        total_frame_scale = opt.get('total_frame_scale', 1)
        total_model_scale = opt.get('total_model_scale', 1)
        opt['total_frame_scale'] = total_frame_scale * m['frame_scale']
        opt['total_model_scale'] = total_model_scale * m['model_scale']

        if model_name in prepared_models.keys():
            continue

        print(f'Load model: {model_name}')
        m['name'] = model_name
        models_module.import_model(m['class'])  # import model class as need
        m['net'] = MODEL_REGISTRY.get(m['class'])(**m['modelargs'])
        if m['need_refresh']:
            m['refresh_func'] = m['net'].refresh

        m['net'] = loadmodel(m['net'], m['pretrain'], opt['NGPUs'])
        m['dataset'] = DATASET_REGISTRY.get(m['dataset_class'])

        scale = m.get('model_scale', 1)
        if m.get('need_pad'):
            m['net'] = functools.partial(forward_pad, forward_function=m['net'],
                                         scale=scale, times=m['need_pad'])
        if opt['chop_forward'] and m.get('chopable'):
            m['net'] = functools.partial(forward_chop, forward_function=m['net'],
                                         scale=scale, n_GPUs=opt['NGPUs'],
                                         multi_output=m['multi_output'], min_size=opt['chop_threshold'])
        prepared_models[model_name] = m

    return prepared_models


def model_forward(model_conf, input_tmpdir, save_tmpdir, opt):
    model = model_conf['net']
    testset = model_conf['dataset'](input_tmpdir.getstring(), n_frames=model_conf['nframes'])
    dataloader = DataLoader(testset, batch_size=opt['batchsize'], shuffle=False, num_workers=4)
    saveimg_function = SAVEIMG_REGISTRY.get(model_conf['saveimg_function'])
    que = queue.Queue(maxsize=100)
    imgsavers = [utils.ImgSaver(saveimg_function, que, f'Saver_{i}') for i in range(opt['num_imgsavers'])]
    for saver in imgsavers:
        saver.start()
    if model_conf['need_refresh']:
        model_conf['refresh_func']()
    with torch.no_grad():
        for lr, filename in tqdm(dataloader):
            lr = lr.to('cuda')
            sr = model(lr).cpu()
            # support models with multi-frame output
            if model_conf['multi_output']:
                sr = sr.transpose(0, 1).reshape(-1,*sr.shape[-3:])
                filename = utils.flatten_list(filename)
            for i in range(sr.shape[0]):
                que.put({'img': sr[i], 'path': save_tmpdir.getPath().joinpath(filename[i])})
    for _ in range(opt['num_imgsavers']):
        que.put('quit')
    for saver in imgsavers:
        saver.join()


def sequential_forward(prepared_models, input_tmpdir, opt):
    save_tmpdir = input_tmpdir # if no models provided, return input
    for model_name in opt['models']:
        save_tmpdir = utils.getTempdir(opt['tempdir_type'], opt)
        print(f'Using model: {model_name}')
        print(f'Input dir: {input_tmpdir}')
        print(f'Save dir: {save_tmpdir}')
        model_forward(prepared_models[model_name], input_tmpdir, save_tmpdir, opt)
        input_tmpdir = save_tmpdir
    return save_tmpdir


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

