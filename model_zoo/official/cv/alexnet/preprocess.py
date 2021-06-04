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
import argparse
import numpy as np
from src.model_utils.config import config
from src.dataset import create_dataset_cifar10
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--dataset_name', type=str, choices=["cifar10", "imagenet2012"], default="cifar10")
parser.add_argument('--data_path', type=str, default='', help='eval data dir')
parser.add_argument("--config_path", type=str, default="../default_config.yaml", help="config file path.")
result_path = './preprocess_Result/'
#parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
args = parser.parse_args()
config.config_path = args.config_path
if __name__ == "__main__":
    if args.dataset_name == "cifar10":
        dataset = create_dataset_cifar10(config, args.data_path, batch_size=config.batch_size, status="eval")
        img_path = os.path.join(result_path, "00_data")
        os.makedirs(img_path)
        label_list = []
        for idx, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
            file_name = "alexnet_data_bs" + str(config.batch_size) + "_" + str(idx) + ".bin"
            file_path = os.path.join(img_path, file_name)
            data["image"].tofile(file_path)
            label_list.append(data["label"])
        np.save(os.path.join(result_path, "cifar10_label_ids.npy"), label_list)
        print("=" * 20, "export bin files finished", "=" * 20)
        