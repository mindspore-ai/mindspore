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
preprocess.
"""
import os
import argparse

import numpy as np
from src.dataset import get_adj_features_labels, get_mask


def generate_bin():
    """Generate bin files."""
    parser = argparse.ArgumentParser(description='preprocess')
    parser.add_argument('--data_dir', type=str, default='./data/cora/cora_mr', help='Dataset directory')
    parser.add_argument('--test_nodes_num', type=int, default=1000, help='Nodes numbers for test')
    parser.add_argument('--result_path', type=str, default='./preprocess_Result/', help='Result path')
    args_opt = parser.parse_args()

    adj, feature, label_onehot, _ = get_adj_features_labels(args_opt.data_dir)
    nodes_num = label_onehot.shape[0]
    test_mask = get_mask(nodes_num, nodes_num - args_opt.test_nodes_num, nodes_num)
    adj_path = os.path.join(args_opt.result_path, "00_data")
    feature_path = os.path.join(args_opt.result_path, "01_data")
    os.makedirs(adj_path)
    os.makedirs(feature_path)
    adj.tofile(os.path.join(adj_path, "adj.bin"))
    feature.tofile(os.path.join(feature_path, "feature.bin"))
    np.save(os.path.join(args_opt.result_path, 'label_onehot.npy'), label_onehot)
    np.save(os.path.join(args_opt.result_path, 'test_mask.npy'), test_mask)

if __name__ == '__main__':
    generate_bin()
