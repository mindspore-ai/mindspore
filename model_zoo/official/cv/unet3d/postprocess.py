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
from src.utils import create_sliding_window, CalculateDice
from src.model_utils.config import config
from src.model_utils.moxing_adapter import moxing_wrapper

@moxing_wrapper()
def cal_acc():
    ''' calculate accuracy'''
    index = 0
    total_dice = 0
    image_path = os.path.join(config.pre_result_path, "image")
    seg_path = os.path.join(config.pre_result_path, "seg")
    file_num = len(os.listdir(image_path))
    for j in range(file_num):
        image = np.load(os.path.join(image_path, "unet3d_bs" + str(config.batch_size) + "_" + str(j) + ".npy"))
        seg = np.load(os.path.join(seg_path, "unet3d_bs" + str(config.batch_size) + "_" + str(j) + ".npy"))
        _, slice_list = create_sliding_window(image, config.roi_size, config.overlap)
        image_size = (config.batch_size, config.num_classes) + image.shape[2:]
        output_image = np.zeros(image_size, np.float32)
        count_map = np.zeros(image_size, np.float32)
        importance_map = np.ones(config.roi_size, np.float32)
        i = 0
        w_name = "unet3d_bs" + str(config.batch_size) + "_bt" + str(j) + "_" + str(i) + "_0.bin"
        w_path = os.path.join(config.post_result_path, w_name)
        while os.path.isfile(w_path):
            pred_shape = (config.batch_size, config.num_classes) + tuple(config.roi_size)
            pred_probs = np.fromfile(w_path, np.float32).reshape(pred_shape)
            slice_ = slice_list[i]
            output_image[slice_] += pred_probs
            count_map[slice_] += importance_map
            i += 1
            w_name = "unet3d_bs" + str(config.batch_size) + "_bt" + str(j) + "_" + str(i) + "_0.bin"
            w_path = os.path.join(config.post_result_path, w_name)

        output_image = output_image / count_map
        dice, _ = CalculateDice(output_image, seg)
        print("The {} batch dice is {}".format(index, dice), flush=True)
        total_dice += dice
        index = index + 1
    eval_data_size = index
    avg_dice = total_dice / eval_data_size
    print("**********************End Eval***************************************")
    print("eval average dice is {}".format(avg_dice))

if __name__ == '__main__':
    cal_acc()
