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
"""pre process for 310 inference"""
import os
import argparse
from src.dataset import create_dataset_cifar

parser = argparse.ArgumentParser(description="squeeze cifar10 preprocess")
parser.add_argument("--dataset_path", type=str, required=True, help="dataset path.")
parser.add_argument("--output_path", type=str, required=True, help="output path.")
args = parser.parse_args()


def preprocess(dataset_path, output_path):
    dataset = create_dataset_cifar(dataset_path, False, batch_size=1)
    img_path = os.path.join(output_path, "img_data")
    label_path = os.path.join(output_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)

    for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        img_data = data["image"]
        img_label = data["label"]
        file_name = "squeeze_cifar10" + "_" + str(idx) + '.bin'
        img_file_path = os.path.join(img_path, file_name)
        img_data.tofile(img_file_path)

        label_file_path = os.path.join(label_path, file_name)
        img_label.tofile(label_file_path)

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    preprocess(args.dataset_path, args.output_path)
