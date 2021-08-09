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
"""preprocess the dataset into h5 files"""
import os
import glob
import argparse as arg
import rawpy
import h5py
import numpy as np


if __name__ == '__main__':
    parser = arg.ArgumentParser(description='data preprocess')
    parser.add_argument('--raw_path', type=str, help='raw data path')
    parser.add_argument('--save_path', type=str, help='save data path')
    args = parser.parse_args()

    raw_in_file_dir = os.path.join(args.raw_path + 'short/')
    raw_gt_file_dir = os.path.join(args.raw_path + 'long/')
    save_path_in = os.path.join(args.save_path + 'short/')
    save_path_gt = os.path.join(args.save_path + 'long/')

    raw_file_paths = glob.glob(raw_in_file_dir + '*.ARW')
    for file_path in raw_file_paths:
        raw = rawpy.imread(file_path)
        im = raw.raw_image_visible.astype(np.float32)
        f = h5py.File(save_path_in + os.path.basename(file_path)[0:-4] + '.hdf5', 'w')
        f.create_dataset('in', data=im)

    raw_file_paths = glob.glob(raw_gt_file_dir + '*.ARW')
    for file_path in raw_file_paths:
        raw = rawpy.imread(file_path)
        im1 = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        f = h5py.File(save_path_gt+os.path.basename(file_path)[0:-4] + '.hdf5', 'w')
        f.create_dataset('gt', data=im1)
