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
from src.dataset import create_tinybert_dataset, DataType


parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument("--eval_data_dir", type=str, default="", help="Data path, it is better to use absolute path")
parser.add_argument("--schema_dir", type=str, default="", help="Schema path, it is better to use absolute path")
parser.add_argument("--dataset_type", type=str, default="tfrecord",
                    help="dataset type tfrecord/mindrecord, default is tfrecord")
parser.add_argument("--result_path", type=str, default="./preprocess_Result/", help="result path")
args_opt = parser.parse_args()

BATCH_SIZE = 32

if args_opt.dataset_type == "tfrecord":
    dataset_type = DataType.TFRECORD
elif args_opt.dataset_type == "mindrecord":
    dataset_type = DataType.MINDRECORD
else:
    raise Exception("dataset format is not supported yet")


def get_bin():
    """
    generate bin files.
    """
    input_ids_path = os.path.join(args_opt.result_path, "00_input_ids")
    token_type_id_path = os.path.join(args_opt.result_path, "01_token_type_id")
    input_mask_path = os.path.join(args_opt.result_path, "02_input_mask")
    label_ids_path = os.path.join(args_opt.result_path, "label_ids.npy")

    os.makedirs(input_ids_path)
    os.makedirs(token_type_id_path)
    os.makedirs(input_mask_path)

    eval_dataset = create_tinybert_dataset('td', batch_size=BATCH_SIZE,
                                           device_num=1, rank=0, do_shuffle="false",
                                           data_dir=args_opt.eval_data_dir,
                                           schema_dir=args_opt.schema_dir,
                                           data_type=dataset_type)
    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    label_list = []
    for j, data in enumerate(eval_dataset.create_dict_iterator(output_numpy=True, num_epochs=1)):
        file_name = "tinybert_bs" + str(BATCH_SIZE) + "_" + str(j) + ".bin"
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        input_ids.tofile(os.path.join(input_ids_path, file_name))
        input_mask.tofile(os.path.join(input_mask_path, file_name))
        token_type_id.tofile(os.path.join(token_type_id_path, file_name))
        label_list.append(label_ids)
    np.save(label_ids_path, label_list)
    print("=" * 20, 'export files finished', "=" * 20)


if __name__ == '__main__':
    get_bin()
