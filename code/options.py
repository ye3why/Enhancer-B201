import yaml
import copy
import os
from collections import OrderedDict
from pathlib import Path
import utils


def parse(args):
    preset_path = './code/settings/presets.yml'
    with open(preset_path, mode='r') as f:
        Loader, _ = utils.ordered_yaml()
        preset = yaml.load(f, Loader=Loader)
    videopresets = preset.pop('presets')

    if 'opt' in args and args.opt:
        # merge preset and --opt
        opt_path = args.opt
        with open(opt_path, mode='r') as f:
            Loader, _ = utils.ordered_yaml()
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
    models_conf = utils.load_ymal(opt['models_conf'])
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
