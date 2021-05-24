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
################preprocess########################
"""
import argparse
import os

from PIL import Image
from src.config import config as cfg

def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('preprocess')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/',
                        help='result path')
    args_opt = parser.parse_args()

    args_opt.data_dir = cfg.data_dir
    args_opt.train_image_dir_name = os.path.join(cfg.data_dir, cfg.train_image_dir_name)
    args_opt.val_fname = cfg.val_fname

    return args_opt

def prepare_valset(arg):
    """generate validate dataset."""
    with open(os.path.join(arg.data_dir, arg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()
    for i, _ in enumerate(f_list):
        item = f_list[i]
        img_filename = str(item).strip().split(',')[0]
        img_path = os.path.join(arg.train_image_dir_name, img_filename)
        img = Image.open(img_path)
        img.save(os.path.join(arg.result_path, img_filename))

if __name__ == '__main__':
    args = parse_args()
    prepare_valset(args)
