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
from src.assessment_method import Accuracy

parser = argparse.ArgumentParser(description="postprocess")
parser.add_argument("--batch_size", type=int, default=1, help="Eval batch size, default is 1")
parser.add_argument("--label_dir", type=str, default="", help="label data dir")
parser.add_argument("--result_dir", type=str, default="./result_Files", help="infer result Files")
parser.add_argument("--dataset", type=str, default="atis", help="dataset")

args, _ = parser.parse_known_args()

if __name__ == "__main__":
    if args.dataset == 'atis':
        num_class = 26
    if args.dataset == 'mrda':
        num_class = 5
    if args.dataset == 'swda':
        num_class = 42
    if args.dataset == 'udc':
        num_class = 2
    callback = Accuracy()
    file_name = os.listdir(args.label_dir)
    for f in file_name:
        f_name = os.path.join(args.result_dir, f.split('.')[0] + '_0.bin')
        logits = np.fromfile(f_name, np.float32).reshape(args.batch_size, num_class)
        logits = Tensor(logits)
        label_ids = np.fromfile(os.path.join(args.label_dir, f), np.int32)
        label_ids = Tensor(label_ids.reshape(args.batch_size, 1))
        callback.update(logits, label_ids)

    print("==============================================================")
    print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                              callback.acc_num / callback.total_num))
    print("==============================================================")
