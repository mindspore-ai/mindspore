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

"""Postprocess for yolov3-resnet18"""
import os
import argparse
import numpy as np
from src.config import ConfigYOLOV3ResNet18
from src.utils import metrics

parser = argparse.ArgumentParser(description='Yolov3 postprocess')
parser.add_argument("--batchsize", type=int, default=1, help="batchsize.")
parser.add_argument("--anno_path", type=str, required=True, help="Annotation path.")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
args = parser.parse_args()

if __name__ == '__main__':
    config = ConfigYOLOV3ResNet18()
    batchsize = args.batchsize

    anno_dict = {}
    for line in open(args.anno_path):
        line_list = line.split(' ')
        line_list[0] = line_list[0].split('/')[-1]
        anno_dict[line_list[0]] = line_list[1:]

    pred_data = []
    for key in anno_dict:
        result0 = os.path.join(args.result_path, key.split('.')[0] + '_0.bin')
        result1 = os.path.join(args.result_path, key.split('.')[0] + '_1.bin')
        output0 = np.fromfile(result0, np.float32).reshape(batchsize, 13860, 4)
        output1 = np.fromfile(result1, np.float32).reshape(batchsize, 13860, 2)

        anno_list = []
        for v in anno_dict[key]:
            v_list = v.split(',')
            anno_list.append(v_list)
        annotation = np.array(anno_list, np.int64)

        for batch_idx in range(batchsize):
            pred_data.append({"boxes": output0[batch_idx],
                              "box_scores": output1[batch_idx],
                              "annotation": annotation})

    precisions, recalls = metrics(pred_data)
    print("\n========================================\n")
    for i in range(config.num_classes):
        print("class {} precision is {:.2f}%, recall is {:.2f}%".format(i, precisions[i] * 100, recalls[i] * 100))
