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
#################lstm postprocess########################
"""
import argparse
import os
import numpy as np
from mindspore.nn import Accuracy
from src.config import lstm_cfg, lstm_cfg_ascend

parser = argparse.ArgumentParser(description='LSTM Postprocess')
parser.add_argument('--label_dir', type=str, default='', help='label data directory.')
parser.add_argument('--result_dir', type=str, default="./result_Files",
                    help='infer result dir.')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['GPU', 'CPU', 'Ascend'],
                    help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
args, _ = parser.parse_known_args()

if __name__ == '__main__':
    metrics = Accuracy()
    rst_path = args.result_dir
    labels = np.load(args.label_dir)

    if args.device_target == 'Ascend':
        cfg = lstm_cfg_ascend
    else:
        cfg = lstm_cfg

    for i in range(len(os.listdir(rst_path))):
        file_name = os.path.join(rst_path, "LSTM_data_bs" + str(cfg.batch_size) + '_' + str(i) + '_0.bin')
        output = np.fromfile(file_name, np.float32).reshape(cfg.batch_size, cfg.num_classes)
        metrics.update(output, labels[i])

    print("result of Accuracy is: ", metrics.eval())
