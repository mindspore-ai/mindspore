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
"""Evaluation api."""
import os
from src.model_utils.config import config
from src.dataset import load_dataset

def generate_bin():
    '''
    Generate bin files.
    '''
    ds = load_dataset(data_files=config.test_dataset,
                      batch_size=config.batch_size,
                      epoch_count=1,
                      sink_mode=config.dataset_sink_mode,
                      shuffle=False) if config.test_dataset else None
    cur_dir = config.result_path
    source_eos_ids_path = os.path.join(cur_dir, "00_source_eos_ids")
    source_eos_mask_path = os.path.join(cur_dir, "01_source_eos_mask")
    target_eos_ids_path = os.path.join(cur_dir, "target_eos_ids")

    if not os.path.isdir(source_eos_ids_path):
        os.makedirs(source_eos_ids_path)
    if not os.path.isdir(source_eos_mask_path):
        os.makedirs(source_eos_mask_path)
    if not os.path.isdir(target_eos_ids_path):
        os.makedirs(target_eos_ids_path)
    for i, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        file_name = "gigaword_bs_" + str(config.batch_size) + "_" + str(i) + ".bin"
        source_eos_ids = data['source_eos_ids']
        source_eos_ids.tofile(os.path.join(source_eos_ids_path, file_name))

        source_eos_mask = data['source_eos_mask']
        source_eos_mask.tofile(os.path.join(source_eos_mask_path, file_name))

        target_eos_ids = data['target_eos_ids']
        target_eos_ids.tofile(os.path.join(target_eos_ids_path, file_name))

    print("="*20, "export bin files finished", "="*20)

if __name__ == '__main__':
    generate_bin()
