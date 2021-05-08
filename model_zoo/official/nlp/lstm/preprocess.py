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
##############preprocess#################
"""
import argparse
import os
import numpy as np
from src.config import lstm_cfg, lstm_cfg_ascend
from src.dataset import lstm_create_dataset

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                    help='path where the pre-process data is stored.')
parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
parser.add_argument('--device_target', type=str, default="Ascend", choices=['GPU', 'CPU', 'Ascend'],
                    help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
args = parser.parse_args()

if __name__ == '__main__':
    if args.device_target == 'Ascend':
        cfg = lstm_cfg_ascend
    else:
        cfg = lstm_cfg

    dataset = lstm_create_dataset(args.preprocess_path, cfg.batch_size, training=False)
    img_path = os.path.join(args.result_path, "00_data")
    os.makedirs(img_path)
    label_list = []
    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "LSTM_data_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = img_path + "/" + file_name
        data['feature'].tofile(file_path)
        label_list.append(data['label'])

    np.save(args.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
