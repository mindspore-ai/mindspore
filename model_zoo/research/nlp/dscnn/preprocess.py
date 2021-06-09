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
import argparse
import numpy as np

from src.config import eval_config
from src.dataset import audio_dataset

def get_bin():
    ''' generate bin files.'''
    parser = argparse.ArgumentParser()
    parser.add_argument('--pre_result_path', type=str, default="preprocess_Result", help='preprocess result path')
    args, model_settings = eval_config(parser)

    test_de = audio_dataset(args.feat_dir, 'testing', model_settings['spectrogram_length'],
                            model_settings['dct_coefficient_count'], args.per_batch_size)

    eval_dataloader = test_de.create_tuple_iterator(output_numpy=True)
    data_path = os.path.join(args.pre_result_path, "00_data")
    os.makedirs(data_path)
    gt_classes_list = []
    i = 0

    for data, gt_classes in eval_dataloader:
        file_name = "dscnn+_bs" + str(args.per_batch_size) + "_" + str(i) + ".bin"
        file_path = os.path.join(data_path, file_name)
        data.tofile(file_path)
        gt_classes_list.append(gt_classes)
        i = i + 1
    np.save(os.path.join(args.pre_result_path, "gt_classes.npy"), gt_classes_list)
    print("=" * 20, "export bin files finished", "=" * 20)

if __name__ == "__main__":
    get_bin()
