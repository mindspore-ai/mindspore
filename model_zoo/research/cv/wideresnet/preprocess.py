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
"""train WideResNet."""
import os
import argparse
from src.dataset import create_dataset

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Ascend WideResnet cifar10 310 preprocess')
    parser.add_argument('--data_path', type=str, required=True, help='Location of data')
    parser.add_argument('--output_path', type=str, required=True, help='Location of output data.')
    parser.add_argument('--device_id', required=True, default=0, help='device_id')
    args = parser.parse_args()

    # create dataset
    dataset = create_dataset(dataset_path=args.data_path, do_train=False,
                             infer_910=False, device_id=args.device_id, batch_size=1)
    step_size = dataset.get_dataset_size()

    img_path = os.path.join(args.output_path, "img_data")
    label_path = os.path.join(args.output_path, "label")
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
