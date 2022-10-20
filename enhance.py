import torch
import torchvision
import threading
import ffmpeg
import tempfile
import os
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

def postprocess(save_dir, input_dir, output_dir, video):
    vf_str = opt['vf_str']
    params = utils.get_ffmpeg_args(video)
    # frame_rate, pix_fmt, bit_rate = params['r'], params['pix_fmt'], params['b:v']
    input_dir_path = Path(input_dir.name) if opt['use_tmpfile_png'] else input_dir
    try:
        (
            ffmpeg
            # .input(str(input_dir_path.joinpath('%d.png')), r=frame_rate)
            .input(str(input_dir_path.joinpath('%d.png')), r=opt['video_spec']['fps'])
            .output(str(output_dir.joinpath(video.name)),
                    vf=vf_str,
                    # pix_fmt=opt['video_spec']['pix_fmt'],
                    # colorspace=opt['video_spec']['colorspace'],
                    # vcodec=opt['video_spec']['vcodec'],
                    video_bitrate=opt['video_spec']['bitrate'],
                    **opt['video_spec']['output_kwargs'])
            .run(quiet=ffmpeg_quiet, overwrite_output=True, cmd=ffmpeg_cmd)
        )
    except ffmpeg.Error as e:
        print(e.stderr.decode())
        raise e
    if opt['use_tmpfile_png']:
        save_dir.cleanup()
        if not opt['keep_png']:
            input_dir.cleanup()
    else:
        shutil.rmtree(save_dir)
        if not opt['keep_png']:
            shutil.rmtree(input_dir)

    with finishedlist_lock:
        finished_list = split_res_dir.joinpath('finished_list.txt')
        with open(finished_list, 'a') as f:
                f.write('{}\n'.format(str(video.name)))

    print('Compressing {} completed.'.format(video.name))

def model_forward(model, model_conf, input_dir, save_dir):
    print('Input dir: {}'.format(input_dir))
    print('Save dir: {}'.format(save_dir))
    testset = model_conf['dataset'](input_dir, n_frames=model_conf['nframes'])
    dataloader = DataLoader(testset, batch_size=opt['batchsize'], shuffle=False, num_workers=4)
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
                                    args=(sr[i], save_dir.joinpath(filename[i]))
                                    )
                threads.append(t)
                t.start()
    for t in threads:
        t.join()
    if opt['use_tmpfile_png']:
        temp_inputdir_holder[idx].cleanup()
        temp_inputdir_holder[idx] = temp_savedir_holder[idx]
        temp_savedir_holder[idx] = (tempfile.TemporaryDirectory())
        save_dir = Path(temp_savedir_holder[idx].name)
        input_dir = Path(temp_inputdir_holder[idx].name)
    else:
        shutil.rmtree(input_dir)
        shutil.move(save_dir, input_dir)
        save_dir.mkdir()
    return input_dir, save_dir

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to demo config Yaml file.')
    parser.add_argument('--models_conf', type=str, default='./models.yml',
                        help='Path to models config Yaml file.')
    args = parser.parse_args()
    opt = options.parse(args.opt)
    models_conf = options.parse_modelsconf(args.models_conf)

    print(f'Model Library: {models_conf.keys()}')

    print(f'Options:')
    print(options.dict2str(opt))


    ffmpeg_quiet = not opt['debug']
    ffmpeg_cmd = opt['ffmpeg_cmd']

    videos_orig = []
    for vp in opt['video_path']:
        if vp.is_dir():
            videos_orig.extend(sorted(list(vp.glob('*'))))
        else:
            videos_orig.append(vp)

    output_dir = opt['output_dir']
    excludes = [v.stem for v in output_dir.glob('*')]
    videos = [v for v in videos_orig if v.stem not in excludes]

    if not output_dir.exists():
        output_dir.mkdir()

    finishedlist_lock = threading.Lock()

    # prepare models
    models = {}
    for model_name in opt['models']:
        print(f'Load model: {model_name}')

        m = copy.deepcopy(models_conf[model_name])
        models_module.import_model(m['class'])  # import model class as need
        m['net'] = MODEL_REGISTRY.get(m['class'])(**m['modelargs'])
        m['net'] = loadmodel(m['net'], m['pretrain'], opt['NGPUs'])
        m['dataset'] = DATASET_REGISTRY.get(m['dataset_class'])

        scale = m.get('model_scale', 1)
        if m.get('need_pad'):
            m['net'] = functools.partial(utils.forward_pad, forward_function=m['net'],
                                         scale=scale, times=m['need_pad'])
        if opt['chop_forward'] and m.get('chopable'):
            m['net'] = functools.partial(utils.forward_chop, forward_function=m['net'],
                                         scale=scale, n_GPUs=opt['NGPUs'],
                                         multi_output=m['multi_output'], min_size=opt['chop_threshold'])

        models[model_name] = m

    for v in set(videos_orig) - set(videos):
        print(f"Video: [ {v} ] already in output directory. Skipped.")

    # process videos
    for idx, video in enumerate(videos):
        print('Processing {}/{}: {}'.format(idx + 1, len(videos), video))

        output_name = video.stem + opt['video_spec']['ext']

        # split video
        split_res_dir = output_dir.joinpath('tmp.split_res_' + video.stem)
        if not split_res_dir.exists():
            split_res_dir.mkdir()
        if opt['use_tmpfile_split']:
            split_dir_tmp = tempfile.TemporaryDirectory()
            split_dir = Path(split_dir_tmp.name)
        else:
            split_dir = output_dir.joinpath('tmp.split_dir_' + video.stem)
            if not split_dir.exists():
                split_dir.mkdir()
        try:
            if opt['segment_method'] == 'hls' or \
            (opt['segment_method'] == 'auto' and utils.get_codec(video) in ['hevc', 'h264']):
                (
                    ffmpeg
                    .input(str(video), vsync=0)
                    .output(str(split_dir.joinpath('{}_{}'.format(video.stem, '.m3u8'))),
                            c='copy', hls_time='15', f='hls')
                    .run(quiet=ffmpeg_quiet, cmd=ffmpeg_cmd)
                )
            else:
                (
                    ffmpeg
                    .input(str(video), vsync=0)
                    .output(str(split_dir.joinpath('{}_%04d{}'.format(video.stem, '.mov'))),
                            map='v:0', c='copy', segment_time='00:00:15',
                            f='segment')
                    .run(quiet=ffmpeg_quiet, cmd=ffmpeg_cmd)
                )
        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise e
        clips = sorted(set.union(set(split_dir.glob('*.ts')), set(split_dir.glob('*.mov'))))

        print('Split dir: {}, {} clips'.format(split_dir, len(clips)))
        print('Split res dir: {}'.format(split_res_dir))

        # excludes = [v.name for v in split_res_dir.glob('*')]
        finished_list = split_res_dir.joinpath('finished_list.txt')
        if Path(finished_list).exists():
            with open(finished_list, 'r') as f:
                excludes = [v.rstrip() for v in f.readlines()]
            clips = [v for v in clips if v.name not in excludes]
            for v in excludes: print(v)
            print(f"{len(excludes)} clips already enhanced. Skipped.")

        # process clips
        threads_compress = []
        temp_savedir_holder = []
        temp_inputdir_holder = []
        input_dir, save_dir = None, None
        for idx, clip in enumerate(clips):
            print('Processing Clip: {} {}/{}'.format(clip.name, idx + 1, len(clips)))
            if opt['use_tmpfile_png']:
                temp_savedir_holder.append(tempfile.TemporaryDirectory())
                temp_inputdir_holder.append(tempfile.TemporaryDirectory())
                save_dir = Path(temp_savedir_holder[idx].name)
                input_dir = Path(temp_inputdir_holder[idx].name)
            else:
                save_dir = output_dir.joinpath('tmp.save_dir_' + clip.stem)
                save_dir.mkdir()
                input_dir = output_dir.joinpath('tmp.input_dir_' + clip.stem)
                input_dir.mkdir()
            try:
                ffmpeg.input(str(clip)).output(str(input_dir.joinpath('%d.png'))).run(quiet=ffmpeg_quiet)
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e


            # for model_name, conf in models.items():
            for model_name in opt['models']:
                conf = models[model_name]
                print(f'Using model: {model_name}')
                input_dir, save_dir = model_forward(conf['net'], conf, input_dir, save_dir)

            if opt['use_tmpfile_png']:
                tc = threading.Thread(target=postprocess,
                                    args=(temp_savedir_holder[idx], temp_inputdir_holder[idx], split_res_dir, clip))
            else:
                tc = threading.Thread(target=postprocess,
                                    args=(save_dir, input_dir, split_res_dir, clip))
            threads_compress.append(tc)
            tc.start()

        for tc in threads_compress:
            tc.join()

        # concat clips
        print('Concatenating clips...')
        split_list = split_res_dir.joinpath('split_list.txt')
        with open(split_list, 'w') as f:
            clips = sorted(split_res_dir.glob('*'))
            clips = [c.name for c in clips if c.name.find('.txt') == -1]
            for c in clips:
                f.write('file {}\n'.format(str(c)))
        # try:
            # (
                # ffmpeg
                # .input(str(split_list), f='concat', safe=0)
                # .output(str(output_dir.joinpath('concat_' + output_name)), c='copy')
                # .run(quiet=ffmpeg_quiet)
            # )
        # except ffmpeg.Error as e:
            # print(e.stderr.decode())
            # raise e

        # copy audio
        if utils.get_ffmpeg_args(video)['has_audio']:
            print('Copying audio...')
            audio_part = ffmpeg.input(str(video)).audio
            # video_part = ffmpeg.input(str(output_dir.joinpath('concat_' + output_name))).video
            video_part = ffmpeg.input(str(split_list), f='concat', safe=0)
            try:
                (
                    ffmpeg
                    .output(audio_part, video_part, str(output_dir.joinpath(output_name)), c='copy')
                    .run(quiet=ffmpeg_quiet, cmd=ffmpeg_cmd)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e
        else:
            try:
                (
                    ffmpeg
                    .input(str(split_list), f='concat', safe=0)
                    .output(str(output_dir.joinpath(output_name)), c='copy')
                    .run(quiet=ffmpeg_quiet, cmd=ffmpeg_cmd)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e
            # shutil.copy(str(output_dir.joinpath('concat_' + output_name)), str(output_dir.joinpath(output_name)))

        # clean up
        if opt['use_tmpfile_split']:
            split_dir_tmp.cleanup()
        else:
            shutil.rmtree(split_dir)
        # os.remove(str(output_dir.joinpath('concat_' + output_name)))
        if not opt['keep_split_res']:
            shutil.rmtree(split_res_dir)

    print('Done.')
