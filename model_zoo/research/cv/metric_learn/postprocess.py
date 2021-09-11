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
from src.utility import recall_topk_parallel
batch_size = 1
parser = argparse.ArgumentParser(description="metric_learn inference")
parser.add_argument("--result_path", type=str, required=True, help="result files path.")
parser.add_argument("--label_path", type=str, required=True, help="image file path.")
args = parser.parse_args()

def cal_acc_sop(result_path, label_path):
    """cal acc"""
    result_shape = (1, 2048)
    files = os.listdir(result_path)
    f, l = [], []
    for file in files:
        full_file_path = os.path.join(result_path, file)
        if os.path.isfile(full_file_path):
            result = np.fromfile(full_file_path, dtype=np.float32).reshape(result_shape)
            label_file = os.path.join(label_path, file.split(".bin")[0][:-2] + ".bin")
            gt = np.fromfile(label_file, dtype=np.int32)
            f.append(result)
            l.append(gt)
    f = np.vstack(f)
    l = np.hstack(l)
    recall = recall_topk_parallel(f, l, k=1)
    print("eval_recall:", recall)

if __name__ == '__main__':
    cal_acc_sop(args.result_path, args.label_path)
