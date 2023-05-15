import threading
import ffmpeg
import shutil
import argparse
import time
from pathlib import Path

import datasets
import save_func
import options
import utils
from model_func import prepare_model, sequential_forward


def main():
    # read options
    opt, models_conf = parseargs()

    # prepare models
    prepared_models = prepare_model(opt, models_conf)

    # resume
    unprocessed_videos = resume(opt)

    utils.ifnot_mkdir(opt['output_dir'])

    print(f'Options:\n', utils.dict2str(opt))

    # process unprocessed videos
    for idx, video_path in enumerate(unprocessed_videos):
        print('Processing {}/{}: {}'.format(idx + 1, len(unprocessed_videos), video_path))
        process_video(video_path, opt, prepared_models)

    print('Done.')

def parseargs():
    parser = argparse.ArgumentParser()
    # option file
    parser.add_argument('--opt', type=str, help='Path to demo config Yaml file.')

    # followings will be ignored if --opt provided
    parser.add_argument('-i', '--input_path', type=str, nargs='+', help='Input video Path or video directory path.')
    parser.add_argument('-o', '--output_dir', type=str, default='./demo/result', help='Output directory.')
    parser.add_argument('--preset', type=str, default='h264', help='Choose video output preset from presets.yml.')
    parser.add_argument('-m', '--model_list', type=str, nargs='+', help='choose models from models.yml')
    # parser.add_argument('--vf_str', type=str, help='Video filter string for ffmpeg, vf=vf_str.')
    parser.add_argument('--chop', action='store_true', help='use chop forward')
    parser.add_argument('--fps', type=int, help='Output FPS.')
    parser.add_argument('--bitrate', type=str, help='Output bitrate.')
    parser.add_argument('--ffmpeg_cmd', type=str, help='ffmpeg command path.')

    # others
    parser.add_argument('--models_conf', type=str, help='Path to models config Yaml file.')
    parser.add_argument('-l', '--list', action='store_true', help='List available models.')
    parser.add_argument('--debug', action='store_true', help='Debug mode.')

    args = parser.parse_args()

    opt, models_conf = options.parse(args)

    if args.list:
        parser.print_help()
        options.print_available_models(models_conf)
        exit(0)

    assert args.opt or args.input_path, 'Please provide input using --opt or --input_path.'
    if args.opt and args.input_path:
        print(f'Use input from {args.opt}, ignore --input_path.')


    return opt, models_conf

def resume(opt):
    videos_orig = []
    for vp in opt['input_path']:
        if vp.is_dir():
            videos_orig.extend(sorted(list(vp.glob('*'))))
        else:
            videos_orig.append(vp)

    excludes = [v.stem for v in opt['output_dir'].glob('*')]
    unprocessed_videos = [v for v in videos_orig if v.stem not in excludes]

    for v in set(videos_orig) - set(unprocessed_videos):
        print(f"Video: [ {v} ] already in output directory. Skipped.")

    return unprocessed_videos


def getvf(opt):
    vf_str = ''
    for vf, vf_args in opt['vfs'].items():
        if vf_args:
            vf = vf + '=' + str(vf_args)
        if vf_str:
            vf_str = vf_str + ',' + vf
        else:
            vf_str = vf
    return vf_str


def compressclip(save_tmpdir, split_res_dir, video_path, finishedlist_lock, opt):
    stime = time.time()
    vf_str = getvf(opt)
    params = utils.get_ffmpeg_args(video_path)
    save_frame_rate = eval(params['r']) * opt['total_frame_scale']
    # frame_rate, pix_fmt, bit_rate = params['r'], params['pix_fmt'], params['b:v']
    try:
        (
            ffmpeg
            # .input(str(input_tmpdir_path.joinpath('%d.png')), r=frame_rate)
            .input(str(save_tmpdir.getPath().joinpath('%8d.png')), r=save_frame_rate)
            .output(str(split_res_dir.joinpath(video_path.name)),
                    vf=vf_str,
                    # pix_fmt=opt['video_spec']['pix_fmt'],
                    # colorspace=opt['video_spec']['colorspace'],
                    # vcodec=opt['video_spec']['vcodec'],
                    video_bitrate=opt['video_spec']['bitrate'],
                    **opt['video_spec']['output_kwargs'])
            .run(quiet=opt['ffmpeg_quiet'], overwrite_output=True, cmd=opt['ffmpeg_cmd'])
        )
    except ffmpeg.Error as e:
        print(e.stderr.decode())
        raise e

    if opt['keep_png_results']:
        shutil.copytree(save_tmpdir.getPath(), split_res_dir.joinpath(video_path.name + '_png_results'))

    with finishedlist_lock:
        finished_list = split_res_dir.joinpath('finished_list.txt')
        with open(finished_list, 'a') as f:
                f.write('{}\n'.format(str(video_path.name)))

    compress_time = time.time() -stime
    print('Compressing {} completed. Used time: {:.3f}s'.format(video_path.name, compress_time))

def process_video(video_path, opt, prepared_models):
    '''
        1. split video to clips
        2. exclude clips already processed
        3. process unprocessed clips
        4. concat clips and copy audio
    '''
    finishedlist_lock = threading.Lock()

    output_name = video_path.stem + opt['video_spec']['ext']

    # split video to clips
    split_res_dir = opt['output_dir'].joinpath('tmp.split_res_' + video_path.stem)
    utils.ifnot_mkdir(split_res_dir)

    split_tmpdir = utils.getTempdir(opt['tempdir_type'], opt)
    try:
        if opt['segment_method'] == 'hls' or \
        (opt['segment_method'] == 'auto' and utils.get_codec(video_path) in ['hevc', 'h264']):
            (
                ffmpeg
                .input(str(video_path), vsync=0)
                .output(str(split_tmpdir.getPath().joinpath('{}_{}'.format(video_path.stem, '.m3u8'))),
                        c='copy', hls_time='15', f='hls')
                .run(quiet=opt['ffmpeg_quiet'], cmd=opt['ffmpeg_cmd'])
            )
        else:
            (
                ffmpeg
                .input(str(video_path), vsync=0)
                .output(str(split_tmpdir.getPath().joinpath('{}_%04d{}'.format(video_path.stem, '.mov'))),
                        map='v:0', c='copy', segment_time='00:00:15',
                        f='segment')
                .run(quiet=opt['ffmpeg_quiet'], cmd=opt['ffmpeg_cmd'])
            )
    except ffmpeg.Error as e:
        print(e.stderr.decode())
        raise e
    clips = sorted(set.union(set(split_tmpdir.getPath().glob('*.ts')), set(split_tmpdir.getPath().glob('*.mov'))))

    print('Split dir: {}, {} clips'.format(split_tmpdir, len(clips)))
    print('Split res dir: {}'.format(split_res_dir))

    # excludes = [v.name for v in split_res_dir.glob('*')]
    finished_list = split_res_dir.joinpath('finished_list.txt')
    if Path(finished_list).exists():
        with open(finished_list, 'r') as f:
            excludes = [v.rstrip() for v in f.readlines()]
        unprocessed_clips = [v for v in clips if v.name not in excludes]
        for v in excludes: print(v)
        print(f"{len(excludes)} clips already enhanced. Skipped.")
    else:
        unprocessed_clips = clips

    # process unprocessed clips
    threads_compress = []

    for idx, clip in enumerate(unprocessed_clips):
        print('Processing Clip: {} {}/{}'.format(clip.name, idx + 1, len(unprocessed_clips)))
        input_tmpdir = utils.getTempdir(opt['tempdir_type'], opt)
        try:
            ffmpeg.input(str(clip)).output(str(input_tmpdir.getPath().joinpath('%8d.png'))).run(quiet=opt['ffmpeg_quiet'])
        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise e

        save_tmpdir = sequential_forward(prepared_models, input_tmpdir, opt)

        tc = threading.Thread(target=compressclip, args=(save_tmpdir, split_res_dir, clip, finishedlist_lock, opt))
        threads_compress.append(tc)
        tc.start()

    for tc in threads_compress:
        tc.join()

    # concat clips and copy audio
    print('Concatenating clips...')
    split_list = split_res_dir.joinpath('split_list.txt')
    with open(split_list, 'w') as f:
        res_clips = sorted(split_res_dir.glob('*'))
        res_clips = [c.name for c in res_clips if c.name.find('.txt') == -1]
        for c in res_clips:
            f.write('file {}\n'.format(str(c)))

    if utils.get_ffmpeg_args(video_path)['has_audio']:
        print('Copying audio...')
        audio_part = ffmpeg.input(str(video_path)).audio
        video_part = ffmpeg.input(str(split_list), f='concat', safe=0)
        try:
            (
                ffmpeg
                .output(audio_part, video_part, str(opt['output_dir'].joinpath(output_name)), c='copy')
                .run(quiet=opt['ffmpeg_quiet'], cmd=opt['ffmpeg_cmd'])
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise e
    else:
        try:
            (
                ffmpeg
                .input(str(split_list), f='concat', safe=0)
                .output(str(opt['output_dir'].joinpath(output_name)), c='copy')
                .run(quiet=opt['ffmpeg_quiet'], cmd=opt['ffmpeg_cmd'])
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise e

    if not opt['keep_clip_results'] and not opt['keep_png_results']:
        shutil.rmtree(split_res_dir)

if __name__ == '__main__':
    main()
