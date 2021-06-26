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
# ===========================================================================
"""preprocess."""
import os
import numpy as np
from src.dataset import audio_dataset
from src.model_utils.config import config


def get_bin():
    ''' generate bin files.'''
    test_de = audio_dataset(config.pre_feat_dir, 'testing', config.model_setting_spectrogram_length,
                            config.model_setting_dct_coefficient_count, config.per_batch_size)

    eval_dataloader = test_de.create_tuple_iterator(output_numpy=True)
    data_path = os.path.join(config.pre_result_path, "00_data")
    os.makedirs(data_path)
    gt_classes_list = []
    i = 0

    for data, gt_classes in eval_dataloader:
        file_name = "dscnn+_bs" + str(config.per_batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(data_path, file_name)
        data.tofile(file_path)
        gt_classes_list.append(gt_classes)
        i = i + 1
    np.save(os.path.join(config.pre_result_path, "gt_classes.npy"), gt_classes_list)
    print("=" * 20, "export bin files finished", "=" * 20)


if __name__ == "__main__":
    get_bin()
