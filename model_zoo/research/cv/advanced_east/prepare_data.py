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
#################train advanced_east on dataset########################
"""
import argparse
import os

from mindspore.common import set_seed
from src.label import process_label, process_label_size

from src.config import config as cfg
from src.dataset import transImage2Mind, transImage2Mind_size
from src.preprocess import preprocess, preprocess_size

set_seed(1)


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('mindspore data prepare')
    parser.add_argument('--device_target', type=str, default='Ascend', choices=['Ascend', 'GPU'],
                        help='device where the code will be implemented. (Default: Ascend)')
    parser.add_argument('--device_id', type=int, default=3, help='device id of GPU or Ascend. (Default: None)')
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    args = parse_args()
    # create data
    preprocess()
    process_label()
    preprocess_size(256)
    process_label_size(256)
    preprocess_size(384)
    process_label_size(384)
    preprocess_size(448)
    process_label_size(448)
    mindrecord_filename = os.path.join(cfg.data_dir, cfg.mindsrecord_train_file)
    transImage2Mind(mindrecord_filename)
    mindrecord_filename = os.path.join(cfg.data_dir, cfg.mindsrecord_test_file)
    transImage2Mind(mindrecord_filename, True)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_train_file_var
    transImage2Mind_size(mindrecord_filename, 256)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_val_file_var
    transImage2Mind_size(mindrecord_filename, 256, True)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_train_file_var
    transImage2Mind_size(mindrecord_filename, 384)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_val_file_var
    transImage2Mind_size(mindrecord_filename, 384, True)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_train_file_var
    transImage2Mind_size(mindrecord_filename, 448)
    mindrecord_filename = cfg.data_dir + cfg.mindsrecord_val_file_var
    transImage2Mind_size(mindrecord_filename, 448, True)
