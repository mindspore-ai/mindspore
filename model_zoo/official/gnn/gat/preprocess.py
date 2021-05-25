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
import argparse
import os

import numpy as np
from src.dataset import load_and_process


def generate_bin():
    """Generate bin files."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/cora/cora_mr', help='Data dir')
    parser.add_argument('--train_nodes_num', type=int, default=140, help='Nodes numbers for training')
    parser.add_argument('--eval_nodes_num', type=int, default=500, help='Nodes numbers for evaluation')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='Result path')
    args = parser.parse_args()

    feature, biases, _, _, _, _, y_test, test_mask = load_and_process(args.data_dir,
                                                                      args.train_nodes_num,
                                                                      args.eval_nodes_num,
                                                                      args.test_nodes_num)
    feature_path = os.path.join(args.result_path, '00_data')
    biases_path = os.path.join(args.result_path, '01_data')
    y_test_path = os.path.join(args.result_path, 'y_test.npy')
    test_mask_path = os.path.join(args.result_path, 'test_mask.npy')

    os.makedirs(feature_path)
    os.makedirs(biases_path)

    feature.tofile(os.path.join(feature_path, 'feature.bin'))
    biases.tofile(os.path.join(biases_path, 'biases.bin'))
    np.save(y_test_path, y_test)
    np.save(test_mask_path, test_mask)

if __name__ == "__main__":
    generate_bin()
