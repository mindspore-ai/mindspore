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
import numpy as np

from src.model_utils.config import config as cfg
from src.dataset import create_dataset


def get_bin():
    '''generate bin files.'''
    ds_eval = create_dataset(cfg.preprocess_path, cfg.batch_size, False)
    img_path = os.path.join(cfg.pre_result_path, "00_feature")
    os.makedirs(img_path)
    label_list = []

    for i, data in enumerate(ds_eval.create_dict_iterator(output_numpy=True)):
        file_name = "textcrnn_bs" + str(cfg.batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(img_path, file_name)

        data["feature"].tofile(file_path)
        label_list.append(data["label"])

    np.save(os.path.join(cfg.pre_result_path, "label_ids.npy"), label_list)
    print("=" * 20, "bin files finished", "=" * 20)

if __name__ == '__main__':
    get_bin()
