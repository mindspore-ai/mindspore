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

'''
postprocess script.
'''

import os
import argparse
import numpy as np
from mindspore import Tensor
from src.assessment_method import Accuracy, F1
from run_ernie_classifier import eval_result_print

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
parser.add_argument("--task_type", type=str, default='chnsenticorp', help="dataset name")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    args.batch_size = 1
    if args.task_type == 'chnsenticorp':
        num_class = 3
        assessment_method = 'accuracy'
        callback = Accuracy()
    elif args.task_type == 'xnli':
        num_class = 3
        assessment_method = 'accuracy'
        callback = Accuracy()
    elif args.task_type == 'dbqa':
        num_class = 2
        assessment_method = 'f1'
        callback = F1(num_class)
    else:
        raise ValueError("dataset not supported, support: [chnsenticorp, xnli, dbqa]")

    file_name = os.listdir(args.label_dir)
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32).reshape(args.batch_size, num_class)
        logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, 1))
        callback.update(logits, label_ids)

    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")
