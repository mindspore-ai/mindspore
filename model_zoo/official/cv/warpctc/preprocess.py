# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
import math as m
from src.model_utils.config import config as cf
from src.dataset import create_dataset

batch_size = 1

if __name__ == "__main__":
    input_size = m.ceil(cf.captcha_height / 64) * 64 * 3
    dataset = create_dataset(dataset_path=cf.dataset_path,
                             batch_size=batch_size,
                             device_target="Ascend")

    img_path = cf.output_path
    if not os.path.isdir(img_path):
        os.makedirs(img_path)
    total = dataset.get_dataset_size()
    iter_num = 0
    label_dict = {}
    for data in dataset.create_dict_iterator(output_numpy=True):
        file_name = str(iter_num) + ".bin"
        img = data["image"]
        label_dict[file_name] = data["label"][0].tolist()
        img.tofile(os.path.join(img_path, file_name))
        iter_num += 1
    with open('./label.txt', 'w') as file:
        for k, v in label_dict.items():
            file.write(str(k) + ',' + str(v) + '\n')
    print("total image num:", total)
