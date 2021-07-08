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
"""unet 310 infer."""
import os
import cv2
import numpy as np

from src.model_utils.config import config
from src.utils import dice_coeff

if __name__ == '__main__':
    rst_path = config.rst_path
    metrics = dice_coeff()
    eval_activate = config.eval_activate.lower()
    if eval_activate not in ("softmax", "argmax"):
        raise ValueError("eval_activate only support 'softmax' or 'argmax'")

    if hasattr(config, "dataset") and config.dataset == "Cell_nuclei":
        img_size = tuple(config.image_size)
        for i, bin_name in enumerate(os.listdir('./preprocess_Result/')):
            f = bin_name.replace(".png", "")
            file_name = rst_path + f + "_0.bin"
            if eval_activate == "softmax":
                rst_out = np.fromfile(file_name, np.float32).reshape(1, 96, 96, 2)
            else:
                rst_out = np.fromfile(file_name, np.int32).reshape(1, 96, 96)
            mask = cv2.imread(os.path.join(config.data_path, f, "mask.png"), cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, img_size)
            mask = mask.astype(np.float32) / 255
            mask = (mask > 0.5).astype(np.int)
            mask = (np.arange(2) == mask[..., None]).astype(int)
            mask = mask.transpose(2, 0, 1).astype(np.float32)
            label = mask.reshape(1, 2, 96, 96)
            metrics.update(rst_out, label)
    else:
        label_list = np.load('label.npy')
        for j in range(len(os.listdir('./preprocess_Result/'))):
            file_name = rst_path + "ISBI_test_bs_1_" + str(j) + "_0" + ".bin"
            if eval_activate == "softmax":
                rst_out = np.fromfile(file_name, np.float32).reshape(1, 576, 576, 2)
            else:
                rst_out = np.fromfile(file_name, np.int32).reshape(1, 576, 576)
            label = label_list[j]
            metrics.update(rst_out, label)

    eval_score = metrics.eval()
    print("============== Cross valid dice coeff is:", eval_score[0])
    print("============== Cross valid IOU is:", eval_score[1])
