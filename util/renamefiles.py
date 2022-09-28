import os
import os.path as osp
import sys
import glob

q = []

q.extend(sys.argv[1:])

while len(q) > 0:
    oldpath = q.pop().rstrip('/')
    newpath = oldpath.replace(' ', '_')
    if oldpath != newpath:
        print(f'mv "{oldpath}" {newpath}')
        os.system(f'mv "{oldpath}" {newpath}')
    if osp.isdir(newpath):
        q.extend(glob.glob(osp.join(newpath, "*")))

