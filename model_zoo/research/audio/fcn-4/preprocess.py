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
'''
preprocess
'''
import os
import numpy as np

from src.model_utils.config import config
from src.dataset import create_dataset

def get_bin():
    """
    generate bin files.
    """
    data_train = create_dataset(config.data_dir, config.val_filename, config.batch_size, ['feature', 'label'],
                                config.num_consumer)
    data_train = data_train.create_tuple_iterator(output_numpy=True)
    res_true = []
    i = 0
    data_path = os.path.join(config.pre_result_path, "00_data")
    os.makedirs(data_path)

    for data, label in data_train:
        file_name = "fcn4_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(data_path, file_name)
        data.tofile(file_path)
        res_true.append(label)
        i = i + 1

    np.save(os.path.join(config.pre_result_path, "label_ids.npy"), res_true)
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == "__main__":
    get_bin()
