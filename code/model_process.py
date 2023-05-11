import torch
import copy
import functools
import queue
from tqdm import tqdm
from torch.utils.data import DataLoader

import utils
import models as models_module
from registry import MODEL_REGISTRY
from registry import DATASET_REGISTRY
from registry import SAVEIMG_REGISTRY

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

        m['net'] = utils.loadmodel(m['net'], m['pretrain'], opt['NGPUs'])
        m['dataset'] = DATASET_REGISTRY.get(m['dataset_class'])

        scale = m.get('model_scale', 1)
        if m.get('need_pad'):
            m['net'] = functools.partial(utils.forward_pad, forward_function=m['net'],
                                         scale=scale, times=m['need_pad'])
        if opt['chop_forward'] and m.get('chopable'):
            m['net'] = functools.partial(utils.forward_chop, forward_function=m['net'],
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

