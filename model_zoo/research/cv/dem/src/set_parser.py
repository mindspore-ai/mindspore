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
Training parameter setting, will be used in train.py
"""
import argparse

def set_parser():
    """parser for train.py and eval.py"""
    parser = argparse.ArgumentParser(description='MindSpore DEMnet Training')
    parser.add_argument('--device_target', type=str, default="Ascend", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: Ascend)')
    parser.add_argument('--device_id', type=int, default=0, help='number of device which is chosen')
    parser.add_argument('--distribute', type=bool, default=False, help='choice of distribute train')
    parser.add_argument('--device_num', type=int, default=1, help='number of device which is used')
    parser.add_argument('--dataset', type=str, default="CUB", choices=['AwA', 'CUB'],
                        help='dataset which is chosen to train (default: AwA)')
    parser.add_argument('--train_mode', type=str, default='att', choices=['att', 'word', 'fusion'],
                        help='mode which is chosen to train (default: attribute)')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size of one step training')
    parser.add_argument('--interval_step', type=int, default=500, help='the interval of printing loss')
    parser.add_argument('--epoch_size', type=int, default=120, help='epoch of training')
    parser.add_argument('--data_path', type=str, default='/data/DEM_data', help='path where the dataset is saved')
    parser.add_argument('--save_ckpt', type=str, default='../output',
                        help='if is test, must provide path where the trained ckpt file')

    parser.add_argument("--file_format", type=str, default="ONNX", choices=["AIR", "ONNX", "MINDIR"], help="export")
    args = parser.parse_args()

    return args
