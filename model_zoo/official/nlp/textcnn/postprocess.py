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
"""
##############postprocess#################
"""
import os
import argparse
import numpy as np
from mindspore.nn.metrics import Accuracy
from src.config import cfg_mr, cfg_subj, cfg_sst2


parser = argparse.ArgumentParser(description='postprocess')
parser.add_argument('--label_dir', type=str, default="", help='label data dir')
parser.add_argument('--result_dir', type=str, default="", help="infer result dir")
parser.add_argument('--dataset', type=str, default="MR", choices=['MR', 'SUBJ', 'SST2'])
args = parser.parse_args()

if __name__ == '__main__':
    if args.dataset == 'MR':
        cfg = cfg_mr
    elif args.dataset == 'SUBJ':
        cfg = cfg_subj
    elif args.dataset == 'SST2':
        cfg = cfg_sst2

    file_prefix = 'textcnn_bs' + str(cfg.batch_size) + '_'

    metric = Accuracy()
    metric.clear()
    label_list = np.load(args.label_dir, allow_pickle=True)

    for idx, label in enumerate(label_list):
        pred = np.fromfile(os.path.join(args.result_dir, file_prefix + str(idx) + '_0.bin'), np.float32)
        pred = pred.reshape(cfg.batch_size, int(pred.shape[0]/cfg.batch_size))
        metric.update(pred, label)
    accuracy = metric.eval()
    print("accuracy: ", accuracy)
