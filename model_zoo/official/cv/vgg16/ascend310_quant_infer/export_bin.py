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
"""generate data and label needed for AIR model inference"""
import os
import sys
import shutil
import numpy as np


def generate_data():
    """
    Generate data and label needed for AIR model inference at Ascend310 platform.
    """
    config.batch_size = 1
    config.image_size = list(map(int, config.image_size.split(',')))
    config.dataset = "cifar10"

    dataset = vgg_create_dataset(config.data_dir, config.image_size, config.batch_size, training=False)
    img_path = os.path.join(config.result_path, "00_data")
    if os.path.exists(img_path):
        shutil.rmtree(img_path)
    os.makedirs(img_path)
    label_list = []
    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "VGG16_data_bs" + str(config.batch_size) + "_" + str(idx) + ".bin"
        file_path = os.path.join(img_path, file_name)
        data["image"].tofile(file_path)
        label_list.append(data["label"])
    np.save(os.path.join(config.result_path, "cifar10_label_ids.npy"), label_list)
    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    sys.path.append("..")
    from src.dataset import vgg_create_dataset
    from model_utils.moxing_adapter import config

    generate_data()
