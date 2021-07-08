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
"""preprocess."""
import argparse
import os

from src.config import ModelConfig
from src.dataset import get_mindrecord_dataset

parser = argparse.ArgumentParser(description='CTR Prediction')
parser.add_argument('--dataset_path', type=str, default="../data/mindrecord", help='Dataset path')
parser.add_argument('--dataset_binary_path', type=str, default="../ascend310/CriteoBinary", help='Checkpoint path')

args = parser.parse_args()

def generate_bin():
    '''generate bin files'''
    config = ModelConfig()
    batch_size = config.batch_size
    ds = get_mindrecord_dataset(args.dataset_path, train_mode=False)
    batch_ids_path = os.path.join(args.dataset_binary_path, "batch_dense")
    batch_wts_path = os.path.join(args.dataset_binary_path, "batch_spare")
    labels_path = os.path.join(args.dataset_binary_path, "batch_labels")

    os.makedirs(batch_ids_path)
    os.makedirs(batch_wts_path)
    os.makedirs(labels_path)

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        file_name = "criteo_bs" + str(batch_size) + "_" + str(i) + ".bin"
        batch_dense = data['cats_vals']
        batch_dense.tofile(os.path.join(batch_ids_path, file_name))

        batch_spare = data['num_vals']
        batch_spare.tofile(os.path.join(batch_wts_path, file_name))

        labels = data['label']
        labels.tofile(os.path.join(labels_path, file_name))

    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == '__main__':
    generate_bin()
