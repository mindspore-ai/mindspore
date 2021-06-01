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

from mindspore import Tensor
from mindspore.common import dtype as mstype
from src.utils import convert_item_id
from src.dataset import TestGraphDataset, load_graph

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, default="Beauty", help="choose which dataset")
parser.add_argument("--datapath", type=str, default="./scripts/data_mr", help="minddata path")
parser.add_argument("--num_neg", type=int, default=10, help="negative sampling rate ")
parser.add_argument("--raw_neighs", type=int, default=40, help="num of sampling neighbors in raw graph")
parser.add_argument("--gnew_neighs", type=int, default=20, help="num of sampling neighbors in sample graph")
parser.add_argument("--result_path", type=str, default="./preprocess_Result/", help="result path")
args = parser.parse_args()



def get_bin():
    """generate bin files."""
    train_graph, _, sampled_graph_list = load_graph(args.datapath)
    test_graph_dataset = TestGraphDataset(train_graph, sampled_graph_list, num_samples=args.raw_neighs,
                                          num_bgcn_neigh=args.gnew_neighs,
                                          num_neg=args.num_neg)

    num_user = train_graph.graph_info()["node_num"][0]
    num_item = train_graph.graph_info()["node_num"][1]

    for i in range(50):
        data_path = os.path.join(args.result_path, "data_" + str(i))
        users_path = os.path.join(data_path, "00_users")
        os.makedirs(users_path)
        items_path = os.path.join(data_path, "01_items")
        os.makedirs(items_path)
        neg_items_path = os.path.join(data_path, "02_neg_items")
        os.makedirs(neg_items_path)
        u_test_neighs_path = os.path.join(data_path, "03_u_test_neighs")
        os.makedirs(u_test_neighs_path)
        u_test_gnew_neighs_path = os.path.join(data_path, "04_u_test_gnew_neighs")
        os.makedirs(u_test_gnew_neighs_path)
        i_test_neighs_path = os.path.join(data_path, "05_i_test_neighs")
        os.makedirs(i_test_neighs_path)
        i_test_gnew_neighs_path = os.path.join(data_path, "06_i_test_gnew_neighs")
        os.makedirs(i_test_gnew_neighs_path)

        test_graph_dataset.random_select_sampled_graph()
        u_test_neighs, u_test_gnew_neighs = test_graph_dataset.get_user_sapmled_neighbor()
        i_test_neighs, i_test_gnew_neighs = test_graph_dataset.get_item_sampled_neighbor()

        u_test_neighs = Tensor(convert_item_id(u_test_neighs, num_user), mstype.int32)
        u_test_gnew_neighs = Tensor(convert_item_id(u_test_gnew_neighs, num_user), mstype.int32)
        i_test_neighs = Tensor(i_test_neighs, mstype.int32)
        i_test_gnew_neighs = Tensor(i_test_gnew_neighs, mstype.int32)

        users = Tensor(np.arange(num_user).reshape(-1,), mstype.int32)
        items = Tensor(np.arange(num_item).reshape(-1,), mstype.int32)
        neg_items = Tensor(np.arange(num_item).reshape(-1, 1), mstype.int32)

        file_name = 'amazon-beauty.bin'
        users.asnumpy().tofile(os.path.join(users_path, file_name))
        items.asnumpy().tofile(os.path.join(items_path, file_name))
        neg_items.asnumpy().tofile(os.path.join(neg_items_path, file_name))
        u_test_neighs.asnumpy().tofile(os.path.join(u_test_neighs_path, file_name))
        u_test_gnew_neighs.asnumpy().tofile(os.path.join(u_test_gnew_neighs_path, file_name))
        i_test_neighs.asnumpy().tofile(os.path.join(i_test_neighs_path, file_name))
        i_test_gnew_neighs.asnumpy().tofile(os.path.join(i_test_gnew_neighs_path, file_name))
    print("=" * 20, "export bin files finished.", "=" * 20)


if __name__ == "__main__":
    get_bin()
