import yaml
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

def parse(opt_path):
    with open(opt_path, mode='r') as f:
        Loader, _ = ordered_yaml()
        opt = yaml.load(f, Loader=Loader)
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    opt['NGPUs'] = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    opt['batchsize'] = opt['batch_size_per_gpu'] * opt['NGPUs']
    opt['video_path'] = [Path(i).expanduser() for i in opt['video_path']]
    opt['output_dir'] = Path(opt['output_dir']).expanduser()

    if 'models' not in opt:
        opt['models'] = {}
    if 'vfs' not in opt:
        opt['vfs'] = {'null': ''}
    if 'debug' not in opt:
        opt['debug'] = False

    # for compatibility
    if 'use_tmpfile_png' not in opt:
        opt['use_tmpfile_png'] = opt['use_tmpfile']
    if 'use_tmpfile_split' not in opt:
        opt['use_tmpfile_split'] = opt['use_tmpfile']

    opt['vf_str'] = ''
    for vf, vf_args in opt['vfs'].items():
        if vf_args:
            vf = vf + '=' + str(vf_args)
        if opt['vf_str']:
            opt['vf_str'] = opt['vf_str']+ ',' + vf
        else:
            opt['vf_str'] = vf


    return opt

def parse_modelsconf(opt_path):
    models_conf = load_ymal(opt_path)
    # set default options
    defult_opt = models_conf.pop('Default')
    for name, opt in models_conf.items():
        for k, v in defult_opt.items():
            if not opt.get(k):
                opt[k] = v

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
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_level * 2) + k + ':['
            msg += dict2str(v, indent_level + 1)
            msg += ' ' * (indent_level * 2) + ']\n'
        else:
            msg += ' ' * (indent_level * 2) + k + ': ' + str(v) + '\n'
    return msg
