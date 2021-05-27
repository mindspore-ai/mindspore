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
"""postprocess."""
import os
import argparse
import numpy as np
from mindspore import Tensor
from src.metrics import AUCMetric

parser = argparse.ArgumentParser(description='postprocess')
parser.add_argument('--result_path', type=str, default="./result_Files", help='result path')
parser.add_argument('--label_path', type=str, default=None, help='label path')
parser.add_argument("--batch_size", type=int, default=16000, help="infer batch size.")
args_opt, _ = parser.parse_known_args()

def get_acc():
    ''' get accuracy '''
    auc_metric = AUCMetric()
    auc_metric.clear()
    batch_size = args_opt.batch_size

    files = os.listdir(args_opt.label_path)

    for f in files:
        rst_file = os.path.join(args_opt.result_path, f.split('.')[0] + '_0.bin')
        label_file = os.path.join(args_opt.label_path, f)

        predict = Tensor(np.fromfile(rst_file, np.float32).reshape(batch_size, 1))
        label = Tensor(np.fromfile(label_file, np.float32).reshape(batch_size, 1))

        res = []
        res.append(predict)
        res.append(predict)
        res.append(label)

        auc_metric.update(*res)

    auc_metric.eval()

if __name__ == '__main__':
    get_acc()
