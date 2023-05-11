import threading
import ffmpeg
import shutil
import argparse
import queue
import time
import yaml
from tqdm import tqdm
from pathlib import Path

import datasets
import options
import utils
from model_utils import  prepare_model, sequential_forward

def main():
    opt, models_conf = parseargs()

    prepared_models = prepare_model(opt, models_conf)

    print(f'Options:\n', options.dict2str(opt))

    for inp_path in opt['input_path']:
        single_img =  inp_path.suffix in ['.jpg', '.bmp', '.png', '.tiff']
        if single_img:
            process_single_img(inp_path, opt, prepared_models)
        else:
            process_clips(inp_path, opt, prepared_models)
    print('Done.')


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--models', type=str, nargs='+', help='choose models from models.yml')
    parser.add_argument('-i', '--input_path', type=str, nargs='+', help='img path or imgs dir')
    parser.add_argument('-o', '--output_dir', type=str, help='output dir')
    parser.add_argument('--chop', action='store_true', help='use chop forward')
    parser.add_argument('--models_conf', type=str, default='./models.yml', help='Path to models config Yaml file.')
    parser.add_argument('--load_weights', type=str, required=False, default=None, help='load specific model weights. Only support single model')
    parser.add_argument('--modelargs', type=yaml.safe_load, required=False, default=None, help='extra model args to instantialize model.Only support single model')
    parser.add_argument('--list', action='store_true', help='List available models.')

    args = parser.parse_args()

    assert  args.input_path, 'Please provide input.'

    opt, models_conf = options.parse(args)

    if args.list:
        parser.print_help()
        print('\nAvailable models:')
        for m in models_conf.keys():
            print('  - ' + m)
        exit(0)

    return opt, models_conf

def process_single_img(inp_path ,opt, prepared_models):
    if opt['output_dir'].joinpath(inp_path.name).exists():
        print(f'Image: {inp_path.name} already in {opt["output_dir"]}. Skipped.')
        return
    if utils.isImg(opt['output_dir']) and opt['output_dir'].exists():
        print(f'Image: {opt["output_dir"]} already exists! Skipped.')
        return

    input_tmpdir  = utils.getTempdir(tempdir_type='mem')
    shutil.copy(inp_path, input_tmpdir.getPath())
    save_tmpdir = sequential_forward(prepared_models, input_tmpdir, opt)
    shutil.copy(save_tmpdir.getPath().joinpath(inp_path.name), opt['output_dir'])

def process_clips(inp_path, opt, prepared_models):
    utils.ifnot_mkdir(opt['output_dir'])

    if len(datasets.glob_pic(inp_path)) > 0:
        clip_paths = [inp_path]
    else:
        # contains several clips
        clip_paths = list(inp_path.glob('*'))


    for clip_path in tqdm(clip_paths, unit='clip'):
        if opt['output_dir'].joinpath(clip_path.name).exists():
            print(f'Clip: {opt["output_dir"].joinpath(clip_path.name)} exists! Skipped.')
            continue

        print(f'Processing {clip_path.name}...')

        input_tmpdir = utils.getTempdir(opt['tempdir_type'], opt)
        shutil.rmtree(input_tmpdir.getPath())
        shutil.copytree(clip_path, input_tmpdir.getPath())

        save_tmpdir = sequential_forward(prepared_models, input_tmpdir, opt)

        shutil.copytree(save_tmpdir.getPath(), opt['output_dir'].joinpath(clip_path.name))


if __name__ == '__main__':
    main()
