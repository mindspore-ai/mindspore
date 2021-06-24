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
##############preprocess cifar-10#################
"""
import ast
import argparse
import os
import numpy as np
from src.dataset import create_dataset

parser = argparse.ArgumentParser(description='preprocess cifar10')
parser.add_argument('--run_distribute', type=ast.literal_eval, default=False, help='Running distributed evaluation.')
parser.add_argument('--dataset_name', type=str, default='cifar10', help='Dataset, Currently only cifar10 is supported.')
parser.add_argument('--eval_dataset_path', type=str, default='./cifar/eval',\
                    help='Dataset path for evaluating SimCLR.')
parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='result path')
parser.add_argument("--batch_size", type=int, default=1, help="batch size")
parser.add_argument('--use_norm', type=ast.literal_eval, default=False, help='Dataset normalize.')
args = parser.parse_args()

if __name__ == '__main__':

    dataset = create_dataset(args, dataset_mode="eval_classifier")
    img_path = os.path.join(args.result_path, "00_data")
    if os.path.exists(img_path):
        os.rmtree(img_path)
    os.makedirs(img_path)
    label_list = []

    for idx, data in enumerate(dataset, start=0):
        _, images, labels = data
        file_name = "cifar10_data_bs" + str(args.batch_size) + "_" + str(idx) + ".bin"
        file_path = img_path + "/" + file_name
        images.asnumpy().tofile(file_path)
        label_list.append(labels.asnumpy())

    np.save(args.result_path + "label_ids.npy", label_list)
    print("="*20, "export bin files finished", "="*20)
