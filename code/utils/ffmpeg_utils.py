import ffmpeg

def get_nb_frames(videopath):
    info = ffmpeg.probe(videopath)
    for stream in info['streams']:
        if stream['codec_type'] == 'video':
            return stream['nb_frames']

def get_codec(video):
    info = ffmpeg.probe(str(video))
    for s in info['streams']:
        if s['codec_type'] == 'video':
            return s['codec_name']

def get_ffmpeg_args(video):
    info = ffmpeg.probe(str(video))
    res = {}
    res['has_audio'] = False
    for s in info['streams']:
        if s['codec_type'] == 'video':
            res['r'] = s['r_frame_rate']
            res['pix_fmt'] = s['pix_fmt']
            res['b:v'] = s.get('bit_rate', info['format']['bit_rate'])
            res['height'] = s['height']
            res['width'] = s['width']
        if s['codec_type'] == 'audio':
            res['has_audio'] = True
    return res
