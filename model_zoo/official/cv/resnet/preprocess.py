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
"""train resnet."""
import os
from src.dataset import create_dataset1 as create_dataset
from src.model_utils.config import config

if __name__ == '__main__':
    # create dataset
    dataset = create_dataset(dataset_path=config.data_path, do_train=False, batch_size=1,
                             target="Ascend")
    step_size = dataset.get_dataset_size()

    img_path = os.path.join(config.output_path, "img_data")
    label_path = os.path.join(config.output_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)

    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]

        file_name = "google_cifar10_1_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)
