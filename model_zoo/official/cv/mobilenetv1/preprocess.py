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
"""preprocess"""
import os
import json
import numpy as np
from src.dataset import create_dataset1

from src.model_utils.config import config


def create_label(result_path, dir_path):
    print("[WARNING] Create imagenet label. Currently only use for Imagenet2012!")
    dirs = os.listdir(dir_path)
    file_list = []
    for file in dirs:
        file_list.append(file)
    file_list = sorted(file_list)

    total = 0
    img_label = {}
    for i, file_dir in enumerate(file_list):
        files = os.listdir(os.path.join(dir_path, file_dir))
        for f in files:
            img_label[f] = i
        total += len(files)

    json_file = os.path.join(result_path, "imagenet_label.json")
    with open(json_file, "w+") as label:
        json.dump(img_label, label)

    print("[INFO] Completed! Total {} data.".format(total))


config.per_batch_size = config.batch_size
#config.image_size = list(map(int, config.image_size.split(',')))


if __name__ == "__main__":
    if config.dataset == "cifar10":
        dataset = create_dataset1(config.data_path, False, config.per_batch_size)
        img_path = os.path.join(config.result_path, "00_data")
        os.makedirs(img_path)
        label_list = []
        for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            file_name = "mobilenetv1_data_bs" + str(config.per_batch_size) + "_" + str(idx) + ".bin"
            file_path = os.path.join(img_path, file_name)
            data["image"].tofile(file_path)
            label_list.append(data["label"])
        np.save(os.path.join(config.result_path, "cifar10_label_ids.npy"), label_list)
        print("=" * 20, "export bin files finished", "=" * 20)
    else:
        create_label(config.result_path, config.data_path)
