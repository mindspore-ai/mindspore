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
Ernie preprocess script.
'''

import os
import argparse
import numpy as np
from mindspore.dataset import MindDataset


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="ernie preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_result/', help='result path')
    parser.add_argument("--dataset", type=str, default="squad", help="dataset")
    args_opt = parser.parse_args()

    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    ds = MindDataset(args.eval_data_file_path,
                     num_parallel_workers=8,
                     shuffle=True)
    input_mask_path = os.path.join(args.result_path, "00_data")
    src_ids_path = os.path.join(args.result_path, "01_data")
    pos_ids_path = os.path.join(args.result_path, "02_data")
    sent_ids_path = os.path.join(args.result_path, "03_data")
    wn_concept_ids_path = os.path.join(args.result_path, "04_data")
    nell_concept_ids_path = os.path.join(args.result_path, "05_data")
    unique_id_path = os.path.join(args.result_path, "06_data")
    os.makedirs(input_mask_path)
    os.makedirs(src_ids_path)
    os.makedirs(pos_ids_path)
    os.makedirs(sent_ids_path)
    os.makedirs(wn_concept_ids_path)
    os.makedirs(nell_concept_ids_path)
    os.makedirs(unique_id_path)

    flag = 1
    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        input_mask = data["input_mask"]
        src_ids = data["src_ids"]
        pos_ids = data["pos_ids"]
        sent_ids = data["sent_ids"]
        wn_concept_ids = data["wn_concept_ids"]
        nell_concept_ids = data["nell_concept_ids"]
        unique_id = data["unique_id"]

        if args.dataset == "squad":
            nell_concept_ids = np.pad(nell_concept_ids, ((0, 0), (0, 3), (0, 0)),
                                      'constant', constant_values=((0, 0), (0, 0), (0, 0)))

        if flag:
            print("input_mask: ", input_mask.shape)
            print("src_ids: ", src_ids.shape)
            print("pos_ids: ", pos_ids.shape)
            print("sent_ids: ", sent_ids.shape)
            print("wn_concept_ids: ", wn_concept_ids.shape)
            print("nell_concept_ids: ", nell_concept_ids.shape)
            print("unique_id: ", unique_id.shape)
            flag = 0

        file_name = args.dataset + "_batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"

        input_mask_file_path = os.path.join(input_mask_path, file_name)
        input_mask.tofile(input_mask_file_path)

        src_ids_file_path = os.path.join(src_ids_path, file_name)
        src_ids.tofile(src_ids_file_path)

        pos_ids_file_path = os.path.join(pos_ids_path, file_name)
        pos_ids.tofile(pos_ids_file_path)

        sent_ids_file_path = os.path.join(sent_ids_path, file_name)
        sent_ids.tofile(sent_ids_file_path)

        wn_concept_file_ids_path = os.path.join(wn_concept_ids_path, file_name)
        wn_concept_ids.tofile(wn_concept_file_ids_path)

        nell_concept_file_ids = os.path.join(nell_concept_ids_path, file_name)
        nell_concept_ids.tofile(nell_concept_file_ids)

        unique_id_file_ids = os.path.join(unique_id_path, file_name)
        unique_id.tofile(unique_id_file_ids)

    print("=" * 20, "export bin files finished", "=" * 20)
