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

import os
import numpy as np
from src.dataset import create_dataset
from src.utils import create_sliding_window
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper


@moxing_wrapper()
def gen_bin(data_path):
    ''' generate bin files.'''
    data_dir = data_path + "/image/"
    seg_dir = data_path + "/seg/"
    eval_dataset = create_dataset(data_path=data_dir, seg_path=seg_dir, is_training=False)
    eval_data_size = eval_dataset.get_dataset_size()
    print("train dataset length is:", eval_data_size)

    window_path = os.path.join(config.pre_result_path, "00_data")
    image_path = os.path.join(config.pre_result_path, "image")
    seg_path = os.path.join(config.pre_result_path, "seg")
    os.makedirs(window_path)
    os.makedirs(image_path)
    os.makedirs(seg_path)
    j = 0

    for batch in eval_dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        image = batch["image"]
        seg = batch["seg"]
        f_name = "unet3d_bs" + str(config.batch_size) + "_" + str(j) + ".npy"
        np.save(os.path.join(image_path, f_name), image)
        np.save(os.path.join(seg_path, f_name), seg)
        sliding_window_list, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        i = 0
        for window, _ in zip(sliding_window_list, slice_list):
            w_name = "unet3d_bs" + str(config.batch_size) + "_bt" + str(j) + "_" + str(i) + ".bin"
            window_file_path = os.path.join(window_path, w_name)
            window.tofile(window_file_path)
            i += 1
        j += 1
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    gen_bin(data_path=config.data_path)
