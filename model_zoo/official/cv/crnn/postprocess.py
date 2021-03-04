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
"""post process for 310 inference"""
import os
import argparse
import numpy as np

from src.metric import CRNNAccuracy
from src.config import config1 as config

parser = argparse.ArgumentParser(description="yolov3_darknet53 inference")
parser.add_argument("--ann_file", type=str, required=True, help="ann file.")
parser.add_argument("--result_path", type=str, required=True, help="image file path.")
parser.add_argument("--dataset", type=str, default="ic03", choices=['ic03', 'ic13', 'svt', 'iiit5k'])
args = parser.parse_args()

def read_annotation(ann_file):
    file = open(ann_file)

    ann = {}
    for line in file.readlines():
        img_info = line.rsplit("/")[-1].split(",")
        img_path = img_info[0].split('/')[-1]
        ann[img_path] = img_info[1].strip()

    return ann

def read_ic13_annotation(ann_file):
    file = open(ann_file)

    ann = {}
    for line in file.readlines():
        img_info = line.split(",")
        img_path = img_info[0].split('/')[-1]
        ann[img_path] = img_info[1].strip().replace('\"', '')

    return ann

def read_svt_annotation(ann_file):
    file = open(ann_file)

    ann = {}
    for line in file.readlines():
        img_info = line.split(",")
        img_path = img_info[0].split('/')[-1]
        ann[img_path] = img_info[1].strip()

    return ann

def get_eval_result(result_path, ann_file):
    """
    Calculate accuracy according to the annotation file and result file.
    """
    metrics = CRNNAccuracy(config)

    if args.dataset == "ic03" or args.dataset == "iiit5k":
        ann = read_annotation(ann_file)
    elif args.dataset == "ic13":
        ann = read_ic13_annotation(ann_file)
    elif args.dataset == "svt":
        ann = read_svt_annotation(ann_file)

    for img_name, label in ann.items():
        result_file = os.path.join(result_path, img_name[:-4] + "_0.bin")
        pred_y = np.fromfile(result_file, dtype=np.float32).reshape(config.num_step, -1, config.class_num)
        metrics.update(pred_y, [label])

    print("result CRNNAccuracy is: ", metrics.eval())
    metrics.clear()

if __name__ == '__main__':
    get_eval_result(args.result_path, args.ann_file)
