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
import os
from src.datasets import create_dataset, DataType
from src.model_utils.config import config


def generate_bin():
    '''generate bin files'''
    data_path = config.dataset_path
    batch_size = config.batch_size
    if config.dataset_type == "tfrecord":
        dataset_type = DataType.TFRECORD
    elif config.dataset_type == "mindrecord":
        dataset_type = DataType.MINDRECORD
    else:
        dataset_type = DataType.H5
    ds = create_dataset(data_path, train_mode=False, epochs=1,
                        batch_size=batch_size, data_type=dataset_type)
    feat_ids_path = os.path.join(config.result_path, "00_feat_ids")
    feat_vals_path = os.path.join(config.result_path, "01_feat_vals")
    label_path = os.path.join(config.result_path, "02_labels")

    os.makedirs(feat_ids_path)
    os.makedirs(feat_vals_path)
    os.makedirs(label_path)

    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True)):
        file_name = "criteo_bs" + str(batch_size) + "_" + str(i) + ".bin"
        batch_ids = data['feat_ids']
        batch_ids.tofile(os.path.join(feat_ids_path, file_name))

        batch_wts = data['feat_vals']
        batch_wts.tofile(os.path.join(feat_vals_path, file_name))

        labels = data['label']
        labels.tofile(os.path.join(label_path, file_name))

    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == '__main__':
    generate_bin()
