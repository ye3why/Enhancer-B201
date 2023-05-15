import os
import os.path as osp
import time
import argparse
import queue
import threading
import functools
import cv2
import numpy as np
import re
from pathlib import Path
import torch
import tqdm
from tabulate import tabulate
from collections import OrderedDict


import utils
import metrics
from registry import METRIC_REGISTRY


def parseargs():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--restored', type=str, help='restored img dir')
    parser.add_argument('-g', '--gt', type=str, help='gt dir')
    parser.add_argument('--suffix', type=str, default='', help='Suffix for restored images')
    parser.add_argument('--output_dir', type=str, default='./evaluations/', help='Output directory')
    parser.add_argument('-m', '--metric_list', type=str, nargs='+', help='choose metrics ')
    parser.add_argument('-l', '--list', action='store_true', help='List available metrics.')
    parser.add_argument('--crop_border', type=int, default=0, help='Crop border for each side')
    parser.add_argument( '--test_y_channel', action='store_true',
        help='If True, test Y channel (In MatLab YCbCr format). If False, test RGB channels.')
    parser.add_argument('--num_workers', type=int, default=3, help='')
    parser.add_argument('--verbose', action='store_true', help='print results.')

    args = parser.parse_args()
    if args.list:
        parser.print_help()
        print('\nAvailable metrics:')
        for m in METRIC_REGISTRY.keys():
            print(f'  - {m}')
        exit(0)

    assert args.restored and args.metric_list, 'Please provide -i/--restored and -m/--metric_list.'

    args.restored = Path(args.restored)
    args.gt = Path(args.gt) if args.gt else None

    return args


def main():
    args = parseargs()

    # output_dir = Path(args.output_dir).joinpath(args.restored.name + '_' + time.strftime('%Y%m%d_%H%M%S', time.localtime()))
    output_dir = Path(args.output_dir).joinpath('_'.join([args.restored.name, *args.metric_list]))
    utils.mkdir_and_rename(output_dir)

    # initialize
    ques = OrderedDict()
    workers = OrderedDict()
    results = OrderedDict()
    for m in args.metric_list:
        if METRIC_REGISTRY.get(m).need_gt():
            assert args.gt, f'--gt ground Truth needed for {m}.'
        ques[m] = queue.Queue(maxsize=10)
        workers[m] = [METRIC_REGISTRY.get(m)(ques[m], f'{m}_{i}', crop_border=args.crop_border, test_y_channel=args.test_y_channel)
                      for i in range(args.num_workers)]
        results[m] = OrderedDict()
        for worker in workers[m]:
            worker.start()

    # gather imgs
    if utils.isImg(args.restored):
        pairs = [[args.restored, args.gt]]
    else:
        pairs, not_founds = check_path(args.restored, args.gt, args.suffix)
        assert len(not_founds) == 0, 'Didn\'t find groud truth for some images. '

    # calculate metrics
    for img_path, gt_path in tqdm.tqdm(pairs):
        img_gt = cv2.imread(str(gt_path), cv2.IMREAD_UNCHANGED).astype(np.float32) if gt_path else None
        img_restored = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED).astype(np.float32)
        for m in args.metric_list:
            ques[m].put({'res': results[m], 'name': img_path, 'restored': img_restored, 'gt': img_gt})

    for m in args.metric_list:
        for _ in range(args.num_workers):
            ques[m].put('quit')
        for worker in workers[m]:
            worker.join()

    # save results
    torch.save(results, output_dir.joinpath('results.pth'))

    for m in args.metric_list:
        total_avg_score = []
        sorted_res = sort_by_dir(results[m])
        writer  = utils.Writer(output_dir.joinpath(args.restored.name + '-' + m + '.txt'), args.verbose)
        for dirname, table in sorted_res.items():
            table = sorted(table, key=lambda x: x[0])
            diraverage =  sum(list(zip(*table))[1]) / len(table)
            table.append(['Average: ', diraverage])
            tablestr = tabulate(table, tablefmt='simple', headers=[dirname, m.upper()], floatfmt='.3f')
            writer.print_and_write(tablestr)
            writer.print_blank_line()
            total_avg_score.append(diraverage)
        if len(total_avg_score) > 1:
            writer.print_and_write(f'Total Average {m.upper()}:\t{sum(total_avg_score)/len(total_avg_score):.3f}\n')
        writer.close()

    print('Done.')

def sort_by_dir(res):
    sorted_res = OrderedDict()
    for img_path, score in sorted(res.items(), key=lambda x: x[0]):
        dirname = img_path.parent.name
        container = sorted_res.get(dirname, [])
        container.append([img_path.name, score])
        sorted_res[dirname] = container
    return sorted_res


def check_path(restored, gt_root=None, suffix=None):
    pairs = []
    not_founds = []
    imgs = sorted(utils.glob_pic(restored, recursive=True))
    if not gt_root:
        return list(zip(imgs, len(imgs)*[None])), []
    for p in imgs:
        if suffix:
            name = p.stem[:-len(suffix)] + p.suffix
        else:
            name = p.name
        gt_p = gt_root.joinpath(*p.parts[len(restored.parts):-1]).joinpath(name)
        pairs.append([p, gt_p])
        if not gt_p.exists():
            print(f'{p} not exists in {gt_root} !')
            not_founds.append(p)
    return pairs, not_founds

if __name__ == "__main__":
    main()
