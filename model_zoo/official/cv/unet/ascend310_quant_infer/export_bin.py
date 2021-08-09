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
"""unet 310 infer preprocess dataset"""
import os
import sys
import shutil
import numpy as np


def generate_data(data_dir, result_path, cross_valid_ind=1):
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """

    _, valid_dataset = create_dataset(data_dir, 1, 1, False, cross_valid_ind, False, do_crop=config.crop,
                                      img_size=config.image_size)
    labels_list = []
    img_path = os.path.join(result_path, "00_data")
    if os.path.exists(img_path):
        shutil.rmtree(img_path)
    os.makedirs(img_path)

    for i, data in enumerate(valid_dataset):
        file_name = "ISBI_test_bs_1_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)
        data[0].asnumpy().tofile(file_path)
        labels_list.append(data[1].asnumpy())
    np.save(os.path.join(result_path, "label.npy"), labels_list)


if __name__ == '__main__':
    sys.path.append("..")
    from src.data_loader import create_dataset
    from src.model_utils.config import config

    generate_data(data_dir=config.data_path, cross_valid_ind=config.cross_valid_ind, result_path=config.result_path)
