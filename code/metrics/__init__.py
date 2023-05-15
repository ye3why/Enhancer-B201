import os
import os.path as osp
import importlib
import glob

arch_folder = osp.dirname(osp.abspath(__file__))
files = glob.glob(osp.join(arch_folder, 'calculate_*.py'))

arch_filenames = [osp.splitext(osp.basename(v))[0] for v in files]
# import all the arch modules
_arch_modules = [importlib.import_module(f'metrics.{file_name}') for file_name in arch_filenames]

