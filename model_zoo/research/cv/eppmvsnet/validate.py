# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""EPP-MVSNet's validation process on BlendedMVS dataset"""

import os
import time
from argparse import ArgumentParser

import cv2
import numpy as np
from tqdm import tqdm

import mindspore.dataset as ds
from mindspore import context
from mindspore.ops import operations as P

from src.eppmvsnet import EPPMVSNet
from src.blendedmvs import BlendedMVSDataset
from src.utils import save_pfm, AverageMeter


def get_opts():
    """set options"""
    parser = ArgumentParser()
    parser.add_argument('--gpu_id', type=int, default=0, choices=[0, 1, 2, 3, 4, 5, 6, 7],
                        help='which gpu used to inference')
    ## data
    parser.add_argument('--root_dir', type=str,
                        default='/home/ubuntu/data/DTU/mvs_training/dtu/',
                        help='root directory of dtu dataset')
    parser.add_argument('--dataset_name', type=str, default='blendedmvs',
                        choices=['blendedmvs'],
                        help='which dataset to train/val')
    parser.add_argument('--split', type=str, default=None,
                        help='which split to evaluate')
    parser.add_argument('--scan', type=str, default=None, nargs='+',
                        help='specify scan to evaluate (must be in the split)')
    # for depth prediction
    parser.add_argument('--n_views', type=int, default=5,
                        help='number of views (including ref) to be used in testing')
    parser.add_argument('--depth_interval', type=float, default=128,
                        help='depth interval unit in mm')
    parser.add_argument('--n_depths', nargs='+', type=int, default=[32, 16, 8],
                        help='number of depths in each level')
    parser.add_argument('--interval_ratios', nargs='+', type=float, default=[4.0, 2.0, 1.0],
                        help='depth interval ratio to multiply with --depth_interval in each level')
    parser.add_argument('--img_wh', nargs="+", type=int, default=[1152, 864],
                        help='resolution (img_w, img_h) of the image, must be multiples of 32')
    parser.add_argument('--ckpt_path', type=str, default='ckpts/exp2/_ckpt_epoch_10.ckpt',
                        help='pretrained checkpoint path to load')
    parser.add_argument('--save_visual', default=False, action='store_true',
                        help='save depth and proba visualization or not')
    parser.add_argument('--entropy_range', action='store_true', default=False,
                        help='whether to use entropy range method')
    parser.add_argument('--conf', type=float, default=0.9,
                        help='min confidence for pixel to be valid')
    parser.add_argument('--levels', type=int, default=3, choices=[3, 4, 5],
                        help='number of FPN levels (fixed to be 3!)')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_opts()
    context.set_context(mode=0, device_target='GPU', device_id=args.gpu_id, save_graphs=False, enable_graph_kernel=True)

    dataset = BlendedMVSDataset(args.root_dir, args.split, n_views=args.n_views, depth_interval=args.depth_interval,
                                img_wh=tuple(args.img_wh), levels=args.levels, scan=args.scan)
    img_wh = args.img_wh
    scans = dataset.scans

    print(args.n_depths)
    print(args.interval_ratios)
    # Step 1. Create depth estimation and probability for each scan
    EPPMVSNet_eval = EPPMVSNet(n_depths=args.n_depths, interval_ratios=args.interval_ratios,
                               entropy_range=args.entropy_range, height=args.img_wh[1], width=args.img_wh[0])
    EPPMVSNet_eval.set_train(False)

    depth_dir = f'results/{args.dataset_name}/{args.split}/depth'
    print('Creating depth and confidence predictions...')
    if args.scan:
        data_range = [i for i, x in enumerate(dataset.metas) if x[0] == args.scan]
    else:
        data_range = range(len(dataset))
    test_loader = ds.GeneratorDataset(dataset, column_names=["imgs", "proj_mats", "init_depth_min", "depth_interval",
                                                             "scan", "vid", "depth_0", "mask_0", "fix_depth_interval"],
                                      num_parallel_workers=1, shuffle=False)
    test_loader = test_loader.batch(batch_size=1)
    test_data_size = test_loader.get_dataset_size()
    print("train dataset length is:", test_data_size)

    pbar = tqdm(enumerate(test_loader.create_tuple_iterator()), dynamic_ncols=True, total=test_data_size)

    metrics = ['stage3_l1_loss', 'stage3_less1_acc', 'stage3_less3_acc']
    avg_metrics = {t: AverageMeter() for t in metrics}

    forward_time_avg = AverageMeter()

    scan_list, vid_list = [], []

    depth_folder = f'{img_wh[0]}_{img_wh[1]}_{args.n_views - 1}'

    for i, sample in pbar:
        imgs, proj_mats, init_depth_min, depth_interval, scan, vid, depth_0, mask_0, fix_depth_interval = sample
        scan = scan[0].asnumpy()
        scan_str = ""
        for num in scan:
            scan_str += chr(num)
        scan = scan_str
        vid = vid[0].asnumpy()

        depth_file_dir = os.path.join(depth_dir, scan, depth_folder)
        if not os.path.exists(depth_file_dir):
            os.makedirs(depth_file_dir, exist_ok=True)

        begin = time.time()

        results = EPPMVSNet_eval(imgs, proj_mats, init_depth_min, depth_interval)

        forward_time = time.time() - begin
        if i != 0:
            forward_time_avg.update(forward_time)

        depth, proba = results
        depth = P.Squeeze()(depth).asnumpy()
        depth = np.nan_to_num(depth)  # change nan to 0
        proba = P.Squeeze()(proba).asnumpy()
        proba = np.nan_to_num(proba)  # change nan to 0

        save_pfm(os.path.join(depth_dir, f'{scan}/{depth_folder}/depth_{vid:04d}.pfm'), depth)
        save_pfm(os.path.join(depth_dir, f'{scan}/{depth_folder}/proba_{vid:04d}.pfm'), proba)

        # record l1 loss of each image
        scan_list.append(scan)
        vid_list.append(vid)

        pred_depth = depth
        gt = P.Squeeze()(depth_0).asnumpy()
        mask = P.Squeeze()(mask_0).asnumpy()

        abs_err = np.abs(pred_depth - gt)
        abs_err_scaled = abs_err / fix_depth_interval.asnumpy()

        l1 = abs_err_scaled[mask].mean()
        less1 = (abs_err_scaled[mask] < 1.).astype(np.float32).mean()
        less3 = (abs_err_scaled[mask] < 3.).astype(np.float32).mean()

        avg_metrics[f'stage3_l1_loss'].update(l1)
        avg_metrics[f'stage3_less1_acc'].update(less1)
        avg_metrics[f'stage3_less3_acc'].update(less3)

        if args.save_visual:
            mi = np.min(depth[depth > 0])
            ma = np.max(depth)
            depth = (depth - mi) / (ma - mi + 1e-8)
            depth = (255 * depth).astype(np.uint8)
            depth_img = cv2.applyColorMap(depth, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/{depth_folder}/depth_visual_{vid:04d}.jpg'), depth_img)
            cv2.imwrite(os.path.join(depth_dir, f'{scan}/{depth_folder}/proba_visual_{vid:04d}.jpg'),
                        (255 * (proba > args.conf)).astype(np.uint8))
        print(f'step {i} time: {forward_time}s')
    print(f'mean forward time: {forward_time_avg.avg}')

    with open(f'results/{args.dataset_name}/{args.split}/metrics.txt', 'w') as f:
        for i in avg_metrics.items():
            f.writelines((i[0]) + ':' + str(np.round(i[1].avg, 4)) + '\n')
        f.writelines('mean forward time(s/pic):' + str(np.round(forward_time_avg.avg, 4)) + '\n')
        f.close()
    print('Done!')
