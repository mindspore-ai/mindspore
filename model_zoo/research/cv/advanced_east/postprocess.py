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
#################postprocess########################
"""
import os
import argparse
import numpy as np
from mindspore import Tensor
from src.config import config as cfg
from src.score import eval_pre_rec_f1


def parse_args(cloud_args=None):
    """parameters"""
    parser = argparse.ArgumentParser('postprocess')
    parser.add_argument('--rst_path', type=str, default='./result_Files/',
                        help='infer result path.')
    args_opt = parser.parse_args()

    args_opt.data_dir = cfg.data_dir
    args_opt.train_image_dir_name = os.path.join(cfg.data_dir, cfg.train_image_dir_name)
    args_opt.val_fname = cfg.val_fname
    args_opt.train_label_dir_name = os.path.join(cfg.data_dir, cfg.train_label_dir_name)
    args_opt.batch_size = 1

    return args_opt


if __name__ == '__main__':
    arg = parse_args()
    obj = eval_pre_rec_f1()
    with open(os.path.join(arg.data_dir, arg.val_fname), 'r') as f_val:
        f_list = f_val.readlines()

    batch_list = np.arange(0, len(f_list), arg.batch_size)
    for idx in batch_list:
        gt_list = []
        for i in range(idx, min(idx + arg.batch_size, len(f_list))):
            item = f_list[i]
            img_filename = str(item).strip().split(',')[0]
            gt_list.append(np.load(os.path.join(arg.train_label_dir_name, img_filename[:-4]) + '.npy'))
        y = np.fromfile(os.path.join(arg.rst_path, img_filename + '_0.bin'), np.float32)
        y = Tensor(y.reshape(1, 7, 112, 112))

        obj.add(y, gt_list)

    print(obj.val())
