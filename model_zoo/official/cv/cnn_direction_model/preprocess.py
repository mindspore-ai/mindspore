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
"""preprocess for cnn"""

import os
import random

import numpy as np
from src.model_utils.config import config
from src.dataset import create_dataset_eval

from mindspore import dataset as de

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

def preprocess():
    dataset_name = config.dataset_name
    dataset_lr, dataset_rl = create_dataset_eval(config.data_root_test +  "/" + dataset_name +
                                                 ".mindrecord0", config=config, dataset_name=dataset_name)

    lr_img_path = os.path.join(config.result_path, "lr_dataset/" + "img_data")
    lr_label_path = os.path.join(config.result_path, "lr_dataset/" + "label")
    os.makedirs(lr_img_path)
    os.makedirs(lr_label_path)

    for idx, data in enumerate(dataset_lr.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]

        file_name = "cnn_fsns_1_" + str(idx) + ".bin"
        img_file_path = os.path.join(lr_img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(lr_label_path, file_name)
        img_label.tofile(label_file_path)

    rl_img_path = os.path.join(config.result_path, "rl_dataset/" + "img_data")
    rl_label_path = os.path.join(config.result_path, "rl_dataset/" + "label")
    os.makedirs(rl_img_path)
    os.makedirs(rl_label_path)

    for idx, data in enumerate(dataset_rl.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]

        file_name = "cnn_fsns_1_" + str(idx) + ".bin"
        img_file_path = os.path.join(rl_img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(rl_label_path, file_name)
        img_label.tofile(label_file_path)

if __name__ == '__main__':
    preprocess()
