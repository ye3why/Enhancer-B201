import ffmpeg
import os
import sys
import threading
import shutil
import tqdm
import tempfile
import argparse
import ffmpeg
from pathlib import Path

import utils
import options


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, required=True, help='Path to demo config Yaml file.')
    args = parser.parse_args()
    opt = options.parse(args.opt)

    print(options.dict2str(opt))

    videos = []
    for vp in opt['video_path']:
        if vp.is_dir():
            videos.extend(sorted(list(vp.glob('*'))))
        else:
            videos.append(vp)
    output_dir = opt['output_dir']

    videos = [output_dir.joinpath(v.stem + opt['video_spec']['ext']) for v in videos]

    for idx, video in enumerate(videos):
        print('Processing {}/{}: {}'.format(idx + 1, len(videos), video))
        output_name = video.stem + opt['video_spec']['ext']
        # split video
        split_res_dir = output_dir.joinpath('tmp.split_res_grain_' + video.stem)
        if not split_res_dir.exists():
            split_res_dir.mkdir()
        split_dir = output_dir.joinpath('tmp.split_dir_grain_' + video.stem)
        if not split_dir.exists():
            split_dir.mkdir()

        try:
            (
                ffmpeg
                .input(str(video))
                .output(str(split_dir.joinpath('{}_%04d{}'.format(video.stem, '.mov'))), map='v:0', c='copy', segment_time='00:00:30', f='segment')
                .run(quiet=True)
            )
        except ffmpeg.Error as e:
            print(e.stderr.decode())
            raise e

        clips = sorted(list(split_dir.glob('*')))
        threads = []

        def filmgrain(clip):
            info = ffmpeg.probe(clip)
            for s in info['streams']:
                if s['codec_type'] == 'video':
                    videoinfo = s
            w = int(videoinfo['width'])
            h = int(videoinfo['height'])
            r = videoinfo['r_frame_rate']
            codec = videoinfo['codec_name']
            biterate = videoinfo['bit_rate']
            # duration = float(info['format']['duration']) - float(info['format']['start_time'])
            duration = float(info['format']['duration'])
            frames = videoinfo['nb_frames']
            try:
                (
                    ffmpeg
                    .input(clip)
                    .output(str(split_res_dir.joinpath(clip.name)),
                            filter_complex=f"\
                                color=black:d={duration}:s={w*2}x{h*2}:r={r}, \
                                geq=lum_expr=random(1)*256:cb=128:cr=128, \
                                deflate=threshold0=15, \
                                dilation=threshold0=10, \
                                eq=contrast=3, \
                                scale={w}x{h} [n]; \
                                [0] eq=saturation=0,geq=lum='0.20*(182-abs(75-lum(X,Y)))':cb=128:cr=128 [o]; \
                                [n][o] blend=c0_mode=multiply,negate [a]; \
                                color=c=black:d={duration}:s={w}x{h}:r={r} [b]; \
                                [0][a] alphamerge [c]; \
                                [b][c] overlay ",
                            tune='grain',
                            **opt['video_spec']['output_kwargs'])
                    .run(quiet=True, overwrite_output=True)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e

        for clip in clips:
            tc = threading.Thread(target=filmgrain, args=(clip,))
            threads.append(tc)
            tc.start()
        for tc in threads:
            tc.join()

        shutil.rmtree(split_dir)

        # concat clips
        print('Concatenating clips...')
        split_list = split_res_dir.joinpath('split_list.txt')
        with open(split_list, 'w') as f:
            clips = sorted(split_res_dir.glob('*'))
            clips = [c.name for c in clips if c.name.find('split_list.txt') == -1]
            for c in clips:
                f.write('file {}\n'.format(str(c)))
        # (
            # ffmpeg
            # .input(str(split_list), f='concat', safe=0).output(str(output_dir.joinpath('concat_' + video.name)), c='copy')
            # .run()
        # )


        # copy audio
        if utils.get_ffmpeg_args(video)['has_audio']:
            print('Copying audio...')
            audio_part = ffmpeg.input(str(video)).audio
            video_part = ffmpeg.input(str(split_list), f='concat', safe=0)
            try:
                (
                    ffmpeg
                    .output(audio_part, video_part, str(output_dir.joinpath(video.stem+ '-final.mov')), c='copy')
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e
        else:
            try:
                (
                    ffmpeg
                    .input(str(split_list), f='concat', safe=0)
                    .output(str(output_dir.joinpath(video.stem+ '-final.mov')), c='copy')
                    .run(quiet=True)
                )
            except ffmpeg.Error as e:
                print(e.stderr.decode())
                raise e
        # if utils.get_ffmpeg_args(video)['has_audio']:
            # audio_part = ffmpeg.input(str(video)).audio
            # video_part = ffmpeg.input(str(output_dir.joinpath('concat_' + video.name))).video
            # (
                # ffmpeg
                # .output(audio_part, video_part, str(output_dir.joinpath(video.stem+ '-final.mov')), c='copy')
                # .run()
            # )
        # else:
            # shutil.copy(str(output_dir.joinpath('concat_' + video.name)), str(output_dir.joinpath(video.stem+ '-final.mov')))
        shutil.rmtree(split_res_dir)
        # os.remove(str(output_dir.joinpath('concat_' + video.name)))

