import os
import os.path as osp
import sys
import glob
def main():
    '''
       recursively get all file paths under provided folders,
       and remove all spaces in these file paths
    '''
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


if __name__ == "__main__":
    main()
