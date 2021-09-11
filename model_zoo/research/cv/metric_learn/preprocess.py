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
"""pre process for 310 inference"""
import os
import argparse
from src.dataset import create_dataset0 as create_dataset
from src.utility import GetDatasetGenerator_eval
parser = argparse.ArgumentParser('preprocess')
parser.add_argument('--data_path', type=str, required=True, help='eval data dir')
parser.add_argument('--result_path', type=str, required=True, help='result path')
args = parser.parse_args()

if __name__ == '__main__':
    VAL_LIST = args.data_path + "/test_half.txt"
    dataset_generator_val = GetDatasetGenerator_eval(args.data_path, VAL_LIST)
    eval_dataset = create_dataset(dataset_generator_val, do_train=False, batch_size=1,
                                  device_num=1, rank_id=0)
    step_size = eval_dataset.get_dataset_size()
    img_path = os.path.join(args.result_path, "img_data")
    label_path = os.path.join(args.result_path, "label")
    os.makedirs(img_path)
    os.makedirs(label_path)
    for idx, data in enumerate(eval_dataset.create_dict_iterator()):
        img_data = data["image"]
        img_label = data["label"]
        file_name = "sop_" + str(idx) + ".bin"
        img_file_path = os.path.join(img_path, file_name)
        img_data.asnumpy().tofile(img_file_path)
        label_file_path = os.path.join(label_path, file_name)
        img_label.asnumpy().tofile(label_file_path)
    print("=" * 20, "export bin files finished", "=" * 20)
