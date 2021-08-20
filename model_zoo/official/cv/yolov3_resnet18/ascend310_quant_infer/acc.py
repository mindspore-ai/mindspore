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
from utils import metrics


parser = argparse.ArgumentParser("yolov3_resnet18 quant postprocess")
parser.add_argument("--anno_path", type=str, required=True, help="path to annotation.npy")
parser.add_argument("--result_path", type=str, required=True, help="path to inference results.")
parser.add_argument("--batch_size", type=int, default=1, help="batch size of data.")
parser.add_argument("--num_classes", type=int, default=2, help="number of classed to detect.")

args, _ = parser.parse_known_args()


def calculate_acc():
    """ Calculate accuracy of yolov3_resnet18 inference"""
    ann = np.load(args.anno_path, allow_pickle=True)
    pred_data = []
    prefix = "Yolov3-resnet18_coco_bs_" + str(args.batch_size) + "_"
    for i in range(len(ann)):
        result0 = os.path.join(args.result_path, prefix + str(i) + "_output_0.bin")
        result1 = os.path.join(args.result_path, prefix + str(i) + "_output_1.bin")
        output0 = np.fromfile(result0, np.float32).reshape(args.batch_size, 13860, 4)
        output1 = np.fromfile(result1, np.float32).reshape(args.batch_size, 13860, 2)
        for batch_idx in range(args.batch_size):
            pred_data.append({"boxes": output0[batch_idx],
                              "box_scores": output1[batch_idx],
                              "annotation": ann[i]})
    precisions, recalls = metrics(pred_data)
    for j in range(args.num_classes):
        print("class {} precision is {:.2f}%, recall is {:.2f}%".format(j, precisions[j] * 100, recalls[j] * 100))


if __name__ == '__main__':
    calculate_acc()
