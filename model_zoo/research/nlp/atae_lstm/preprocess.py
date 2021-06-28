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
"""
##############preprocess#################
"""
import argparse
import os

from src.config import config
from src.load_dataset import load_dataset

parser = argparse.ArgumentParser(description='preprocess')
parser.add_argument('--result_path', type=str,
                    default='./preprocess_Result/', help='result path')
parser.add_argument('--device_target', type=str, default="Ascend",
                    choices=['GPU', 'CPU', 'Ascend'],
                    help='the target device to run, support "GPU", "CPU". Default: "Ascend".')
args = parser.parse_args()

if __name__ == '__main__':

    dataset = load_dataset(input_files=config.test_dataset,
                           batch_size=1)

    content_path = os.path.join(args.result_path, "00_content")
    sen_len_path = os.path.join(args.result_path, "01_sen_len")
    aspect_path = os.path.join(args.result_path, "02_aspect")
    solution_path = os.path.join(args.result_path, "solution_path")
    if not os.path.isdir(content_path):
        os.makedirs(content_path)
    if not os.path.isdir(sen_len_path):
        os.makedirs(sen_len_path)
    if not os.path.isdir(aspect_path):
        os.makedirs(aspect_path)
    if not os.path.isdir(solution_path):
        os.makedirs(solution_path)

    for i, data in enumerate(dataset.create_dict_iterator(output_numpy=True)):
        file_name = "atae_lstm_bs" + str(config.batch_size) + "_" + str(i) + ".bin"
        content = data['content']
        content.tofile(os.path.join(content_path, file_name))
        sen_len = data['sen_len']
        sen_len.tofile(os.path.join(sen_len_path, file_name))
        aspect = data['aspect']
        aspect.tofile(os.path.join(aspect_path, file_name))
        solution = data['solution']
        solution.tofile(os.path.join(solution_path, file_name))

    print("=" * 10, "export bin files finished", "=" * 10)
