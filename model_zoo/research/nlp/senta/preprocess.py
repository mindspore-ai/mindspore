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
from mindspore.dataset import MindDataset


def parse_args():
    """set and check parameters."""
    parser = argparse.ArgumentParser(description="ernie preprocess")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--eval_batch_size", type=int, default=1, help="Eval batch size, default is 1")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')

    args_opt = parser.parse_args()

    if args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")
    return args_opt


if __name__ == "__main__":
    args = parse_args()
    ds = MindDataset(
        args.eval_data_file_path,
        num_parallel_workers=8,
        shuffle=True)
    ids_path = os.path.join(args.result_path, "00_data")
    mask_path = os.path.join(args.result_path, "02_data")
    token_path = os.path.join(args.result_path, "01_data")
    label_path = os.path.join(args.result_path, "03_data")
    os.makedirs(ids_path)
    os.makedirs(mask_path)
    os.makedirs(token_path)
    os.makedirs(label_path)

    for idx, data in enumerate(ds.create_dict_iterator(output_numpy=True, num_epochs=1)):
        input_ids = data["src_ids"]
        input_mask = data["mask_ids"]
        token_type_id = data["sent_ids"]
        label_ids = data["label"]

        file_name = "senta_batch_" + str(args.eval_batch_size) + "_" + str(idx) + ".bin"
        ids_file_path = os.path.join(ids_path, file_name)
        input_ids.tofile(ids_file_path)

        mask_file_path = os.path.join(mask_path, file_name)
        input_mask.tofile(mask_file_path)

        token_file_path = os.path.join(token_path, file_name)
        token_type_id.tofile(token_file_path)

        label_file_path = os.path.join(label_path, file_name)
        label_ids.tofile(label_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
