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
"""eval vot"""

import argparse
import os
import json
import sys
import time
import numpy as np
from tqdm import tqdm

from mindspore import context
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from src import evaluation as eval_
from src.net import SiameseRPN
from src.tracker import SiamRPNTracker

import cv2

sys.path.append(os.getcwd())


def get_axis_aligned_bbox(region):
    """ convert region to (cx, cy, w, h) that represent by axis aligned box
    """
    nv = len(region)
    region = np.array(region)
    if nv == 8:
        x1 = min(region[0::2])
        x2 = max(region[0::2])
        y1 = min(region[1::2])
        y2 = max(region[1::2])
        A1 = np.linalg.norm(region[0:2] - region[2:4]) * \
            np.linalg.norm(region[2:4] - region[4:6])
        A2 = (x2 - x1) * (y2 - y1)
        s = np.sqrt(A1 / A2)
        w = s * (x2 - x1) + 1
        h = s * (y2 - y1) + 1
        x = x1
        y = y1
    else:
        x = region[0]
        y = region[1]
        w = region[2]
        h = region[3]

    return x, y, w, h


def test(model_path, data_path, save_name):
    """ using tracking """
    # ------------ prepare data  -----------
    direct_file = os.path.join(data_path, 'list.txt')
    with open(direct_file, 'r') as f:
        direct_lines = f.readlines()
    video_names = np.sort([x.split('\n')[0] for x in direct_lines])
    video_paths = [os.path.join(data_path, x) for x in video_names]
    # ------------ prepare models  -----------
    model = SiameseRPN()
    param_dict = load_checkpoint(model_path)
    param_not_load = load_param_into_net(model, param_dict)
    print(param_not_load)
    # ------------ starting validation  -----------
    results = {}
    accuracy = 0
    all_overlaps = []
    all_failures = []
    gt_lenth = []
    for video_path in tqdm(video_paths, total=len(video_paths)):
        # ------------ prepare groundtruth  -----------
        groundtruth_path = os.path.join(video_path, 'groundtruth.txt')
        with open(groundtruth_path, 'r') as f:
            boxes = f.readlines()
        if ',' in boxes[0]:
            boxes = [list(map(float, box.split(','))) for box in boxes]
        else:
            boxes = [list(map(int, box.split())) for box in boxes]
        gt = boxes.copy()
        gt[:][2] = gt[:][0] + gt[:][2]
        gt[:][3] = gt[:][1] + gt[:][3]
        frames = [os.path.join(video_path, 'color', x) for x in np.sort(os.listdir(os.path.join(video_path, '/color')))]
        frames = [x for x in frames if '.jpg' in x]
        tic = time.perf_counter()
        template_idx = 0
        tracker = SiamRPNTracker(model)
        res = []
        for idx, frame in tqdm(enumerate(frames), total=len(frames)):
            frame = cv2.imdecode(np.fromfile(frame, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            h, w = frame.shape[0], frame.shape[1]
            if idx == template_idx:
                box = get_axis_aligned_bbox(boxes[idx])
                tracker.init(frame, box)
                res.append([1])
            elif idx < template_idx:
                res.append([0])
            else:
                bbox, _ = tracker.update(frame)
                bbox = np.array(bbox)
                bbox = list((bbox[0] - bbox[2] / 2 + 1 / 2, bbox[1] - bbox[3] / 2 + 1 / 2, \
                             bbox[0] + bbox[2] / 2 - 1 / 2, bbox[1] + bbox[3] / 2 - 1 / 2))
                if eval_.judge_failures(bbox, boxes[idx], 0):
                    res.append([2])
                    print('fail')
                    template_idx = min(idx + 5, len(frames) - 1)
                else:
                    res.append(bbox)
        duration = time.perf_counter() - tic
        acc, overlaps, failures, num_failures = eval_.calculate_accuracy_failures(res, gt, [w, h])
        accuracy += acc
        result1 = {}
        result1['acc'] = acc
        result1['num_failures'] = num_failures
        result1['fps'] = round(len(frames) / duration, 3)
        results[video_path.split('/')[-1]] = result1
        all_overlaps.append(overlaps)
        all_failures.append(failures)
        gt_lenth.append(len(frames))
    all_length = sum([len(x) for x in all_overlaps])
    robustness = sum([len(x) for x in all_failures]) / all_length * 100
    eao = eval_.calculate_eao("VOT2015", all_failures, all_overlaps, gt_lenth)
    result1 = {}
    result1['accuracy'] = accuracy / float(len(video_paths))
    result1['robustness'] = robustness
    result1['eao'] = eao
    results['all_videos'] = result1
    print('accuracy is ', accuracy / float(len(video_paths)))
    print('robustness is ', robustness)
    print('eao is ', eao)
    json.dump(results, open(save_name, 'w'))

def parse_args():
    '''parse_args'''
    parser = argparse.ArgumentParser(description='Mindspore SiameseRPN Infering')
    parser.add_argument('--platform', type=str, default='Ascend', choices=('Ascend'), help='run platform')
    parser.add_argument('--device_id', type=int, default=0, help='DEVICE_ID')
    parser.add_argument('--dataset_path', type=str, default='', help='Dataset path')
    parser.add_argument('--checkpoint_path', type=str, default='', help='checkpoint of siamRPN')
    parser.add_argument('--filename', type=str, default='', help='save result file')
    args_opt = parser.parse_args()
    return args_opt

if __name__ == '__main__':
    args = parse_args()
    if args.platform == 'Ascend':
        device_id = args.device_id
        context.set_context(device_id=device_id)
    context.set_context(mode=context.GRAPH_MODE, device_target=args.platform)
    model_file_path = args.checkpoint_path
    data_file_path = args.dataset_path
    save_file_name = args.filename
    test(model_path=model_file_path, data_path=data_file_path, save_name=save_file_name)
