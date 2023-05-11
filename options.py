import yaml
import copy
import os
from collections import OrderedDict
from os import path as osp
from pathlib import Path
from utils import get_ffmpeg_args


def ordered_yaml():
    """Support OrderedDict for yaml.

    Returns:
        yaml Loader and Dumper.
    """
    try:
        from yaml import CDumper as Dumper
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Dumper, Loader

    _mapping_tag = yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG

    def dict_representer(dumper, data):
        return dumper.represent_dict(data.items())

    def dict_constructor(loader, node):
        return OrderedDict(loader.construct_pairs(node))

    Dumper.add_representer(OrderedDict, dict_representer)
    Loader.add_constructor(_mapping_tag, dict_constructor)
    return Loader, Dumper

def load_ymal(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    return opt

def parse(args):
    preset_path = './options/presets.yml'
    with open(preset_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        preset = yaml.load(f, Loader=Loader)
    videopresets = preset.pop('presets')

    if 'opt' in args and args.opt:
        # merge preset and --opt
        opt_path = args.opt
        with open(opt_path, mode='r') as f:
            Loader, _ = ordered_yaml()
            opt = yaml.load(f, Loader=Loader)
        for k,v in preset.items():
            if k not in opt.keys():
                opt[k] = v
    else:
        # no --opt, merge argments from args.
        opt = copy.deepcopy(preset)
        for k, v in vars(args).items():
            if v:
                opt[k] = v

        if 'preset'in args and args.preset:
            assert args.preset in videopresets.keys(), f'{args.preset} not in presets.yml.'
            opt['video_spec'] = videopresets[args.preset]


    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    opt['NGPUs'] = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    opt['batchsize'] = opt['batch_size_per_gpu'] * opt['NGPUs']
    opt['input_path'] = [Path(i).expanduser() for i in opt['input_path']]
    opt['output_dir'] = Path(opt['output_dir']).expanduser()

    opt['ffmpeg_quiet'] =  not opt['debug']

    models_conf = parse_modelsconf(opt)

    return opt, models_conf

def parse_modelsconf(opt):
    models_conf = load_ymal(opt['models_conf'])
    # set default options
    defult_opt = models_conf.pop('Default')
    for name, m in models_conf.items():
        for k, v in defult_opt.items():
            if not m.get(k):
                m[k] = v

        if 'load_weights' in opt and opt['load_weights']:
            m['pretrain'] = opt['load_weights']
        if 'modelargs' in opt:
            m['modelargs'].update(opt['modelargs'])

    return models_conf



def dict2str(opt, indent_level=1):
    """dict to string for printing options.

    Args:
        opt (dict): Option dict.
        indent_level (int): Indent level. Default: 1.

    Return:
        (str): Option string for printing.
    """
    msg = '\n'
    for k in sorted(opt.keys()):
        if isinstance(opt[k], dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(opt[k], indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(opt[k]) + '\n'
    return msg
