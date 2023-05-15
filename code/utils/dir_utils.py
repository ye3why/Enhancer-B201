import string
import shutil
import random
import tempfile
import os
import time
from pathlib import Path
import re

def glob_pic(dirname, recursive=False):
    pics = []
    imgs = Path(dirname).rglob('*') if recursive else Path(dirname).glob('*')
    for x in imgs:
        p = re.search('.*\.(jpe?g|png|bmp|tif)$', str(x), flags=re.IGNORECASE)
        if p is not None:
            pics.append(Path(p.group()))
    return pics

def ifnot_mkdir(path):
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.
    """
    path = Path(path)
    if path.exists():
        new_name = path.name + '_archived_' + get_time_str()
        print(f'Path {path} already exists. Rename it to {new_name}', flush=True)
        os.rename(path, path.parent.joinpath(new_name))
    path.mkdir(parents=True)
    # os.makedirs(path, exist_ok=True)


def isImg(path):
    if isinstance(path, str):
        path = Path(path)
    return path.suffix in ['.jpg', '.bmp', '.png', '.tiff']


def getTempdir(tempdir_type, opt=None):
    if tempdir_type == 'mem':
        return MemoryTempDir()
    elif tempdir_type == 'disk':
        return DiskTempDir(opt)
    else:
        raise NotImplementedError

class MemoryTempDir():
    def __init__(self):
        self.handler = tempfile.TemporaryDirectory()
        self.Path = Path(self.handler.name)
        self.string = self.handler.name

    def __del__(self):
        self.handler.cleanup()

    def __str__(self):
        return self.string

    def getPath(self):
        return self.Path

    def getstring(self):
        return self.string


class DiskTempDir():
    def __init__(self, opt):
        randomname = ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase + string.digits, 10))
        self.handler = opt['output_dir'].joinpath('_'.join(['tmp', randomname]))
        assert not self.handler.exists(), f'tempdir path: {self.handler.name} exists!'
        self.handler.mkdir()
        self.Path = self.handler
        self.string = str(self.handler)

    def __del__(self):
        shutil.rmtree(self.handler)

    def __str__(self):
        return self.string

    def getPath(self):
        return self.Path

    def getstring(self):
        return self.string
