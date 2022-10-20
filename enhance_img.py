import torch
import torchvision
import threading
import ffmpeg
import tempfile
import cv2
import os
import os.path as osp
import copy
import shutil
import argparse
import functools
from pathlib import Path
from tqdm import tqdm
from collections import OrderedDict
from torch.utils.data import DataLoader

import options
import utils
import datasets
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

def model_forward(model, model_conf, input_dir, save_dir):
    print('Input dir: {}'.format(input_dir))
    print('Save dir: {}'.format(save_dir))
    testset = model_conf['dataset'](input_dir, n_frames=model_conf['nframes'])
    dataloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=4)
    saveimg_function = SAVEIMG_REGISTRY.get(model_conf['saveimg_function'])
    threads = []
    with torch.no_grad():
        for lr, filename in tqdm(dataloader):
            lr = lr.to('cuda')
            aux_input = model_conf.get('aux_input', {})
            sr = model(lr, **aux_input)
            # support models with multi-frame output
            if model_conf['multi_output']:
                sr = sr.transpose(0, 1).reshape(-1,*sr.shape[-3:])
                filename = utils.flatten_list(filename)
            for i in range(sr.shape[0]):
                t = threading.Thread(target=saveimg_function,
                                    args=(sr[i], osp.join(save_dir, filename[i]))
                                    )
                threads.append(t)
                t.start()
    for t in threads:
        t.join()

def model_forward_single(model, model_conf, input_dir, save_dir):
    transformer = torchvision.transforms.ToTensor()
    lr = cv2.imread(str(input_dir), cv2.IMREAD_ANYCOLOR|cv2.IMREAD_ANYDEPTH)
    lr = lr[:, :, [2, 1, 0]]
    lr = transformer(lr) # [0.0, 1.0]
    lr = lr.unsqueeze(0)
    lr = lr.to('cuda')
    filename = input_dir.name
    if single_img:
        output_path = save_dir
    else:
        output_path = save_dir.joinpath(filename)

    print('Input Image: {}'.format(input_dir))
    print('Output Image: {}'.format(output_path))
    saveimg_function = SAVEIMG_REGISTRY.get(model_conf['saveimg_function'])
    with torch.no_grad():
            aux_input = model_conf.get('aux_input', {})
            sr = model(lr, **aux_input)
    saveimg_function(sr[0], output_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_list', type=str, required=True, nargs='+', help='choose models from models.yml')
    parser.add_argument('-i', '--input_path', type=str, required=True, help='img path or imgs dir')
    parser.add_argument('-o', '--output_dir', type=str, required=True, help='output dir')
    parser.add_argument('--chop', action='store_true', help='use chop forward')
    parser.add_argument('--not_usetmp', action='store_false', help='not use tempfile')
    parser.add_argument('--NGPUs', type=int, default=1, help='the number of gpus')
    parser.add_argument('--models_conf', type=str, default='./models.yml', help='Path to models config Yaml file.')
    args = parser.parse_args()
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    single_img =  input_path.suffix in ['.jpg', '.bmp', '.png', '.tiff']
    if single_img:
        assert not output_dir.joinpath(input_path.name).exists(), f'{input_path.name} already in {output_dir}'
        if output_dir.suffix in ['.jpg', '.bmp', '.png', '.tiff']:
            assert not output_dir.exists(), f'{output_dir} already exists!'

    else:
        assert not output_dir.exists(), f'Output path {output_dir} exists!'
    # if not single_img and not output_dir.exists():
        # output_dir.mkdir()

    input_dir = tempfile.TemporaryDirectory()
    if input_path.is_dir():
        shutil.rmtree(input_dir.name)
        shutil.copytree(input_path, input_dir.name)
    elif input_path.is_file():
        shutil.copy(input_path, input_dir.name)
    else:
        print('Error: input_path is not a file or a directory.')

    models_conf = options.parse_modelsconf(args.models_conf)
    for m_idx in range(len(args.model_list)):
        model_name = args.model_list[m_idx]
        m = copy.deepcopy(models_conf[model_name])
        models_module.import_model(m['class'])  # import model class as need
        m['net'] = MODEL_REGISTRY.get(m['class'])(**m['modelargs'])
        m['net'] = loadmodel(m['net'], m['pretrain'], args.NGPUs)
        m['dataset'] = DATASET_REGISTRY.get(m['dataset_class'])

        scale = m.get('model_scale', 1)
        if m.get('need_pad'):
            m['net'] = functools.partial(utils.forward_pad, forward_function=m['net'],
                                            scale=scale, times=m['need_pad'])
        if args.chop and m.get('chopable'):
            m['net'] = functools.partial(utils.forward_chop, forward_function=m['net'],
                                            scale=scale, n_GPUs=args.NGPUs,
                                            multi_output=m['multi_output'], min_size=160000)

        saveimg_function = SAVEIMG_REGISTRY.get(m['saveimg_function'])
        save_dir = tempfile.TemporaryDirectory()
        print(f'Using model: {model_name}')
        model_forward(m['net'], m, input_dir.name, save_dir.name)
        input_dir.cleanup()
        input_dir = save_dir

    if input_path.is_dir():
        # model_forward(m['net'], m, input_dir, save_dir)
        shutil.copytree(input_dir.name, output_dir)
    elif input_path.is_file():
        shutil.copy(osp.join(input_dir.name, osp.basename(input_path)), output_dir)
        # model_forward_single(m['net'], m, input_dir, save_dir)
    input_dir.cleanup()
    print('Done.')
