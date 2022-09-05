import importlib
import ast
from copy import deepcopy
from os import path as osp
import glob

# automatically scan and import arch modules for registry
# scan all the files under the 'archs' folder and collect files ending with
# '_arch.py'
arch_folder = osp.dirname(osp.abspath(__file__))
files = glob.glob(osp.join(arch_folder, '*_arch.py'))

d = dict()  # correspondence between classname and filename
for file in files:
    with open(file) as f:
        node = ast.parse(f.read())
    for n in node.body:
        if isinstance(n, ast.ClassDef):
            d[n.name] = osp.splitext(osp.basename(file))[0]

# arch_filenames = [osp.splitext(osp.basename(v))[0] for v in files]
# import all the arch modules
# _arch_modules = [importlib.import_module(f'models.{file_name}') for file_name in arch_filenames]


def import_model(model_class):
    filename = d[model_class]
    importlib.import_module(f'models.{filename}')
