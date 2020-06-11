# Copyright 2020 Huawei Technologies Co., Ltd
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
create adjacency matrix, node features, labels, and mask for training.
"""
import numpy as np
import scipy.sparse as sp
import mindspore.dataset as ds


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def get_adj_features_labels(data_dir):
    """Get adjacency matrix, node features and labels from dataset."""
    g = ds.GraphData(data_dir)
    nodes = g.get_all_nodes(0)
    nodes_list = nodes.tolist()
    row_tensor = g.get_node_feature(nodes_list, [1, 2])
    features = row_tensor[0]
    labels = row_tensor[1]

    nodes_num = labels.shape[0]
    class_num = labels.max() + 1
    labels_onehot = np.eye(nodes_num, class_num)[labels].astype(np.float32)

    neighbor = g.get_all_neighbors(nodes_list, 0)
    node_map = {node_id: index for index, node_id in enumerate(nodes_list)}
    adj = np.zeros([nodes_num, nodes_num], dtype=np.float32)
    for index, value in np.ndenumerate(neighbor):
        # The first column of neighbor is node_id, second column to last column are neighbors of the first column.
        # So we only care index[1] > 1.
        # If the node does not have that many neighbors, -1 is padded. So if value < 0, we will not deal with it.
        if value >= 0 and index[1] > 0:
            adj[node_map[neighbor[index[0], 0]], node_map[value]] = 1
    adj = sp.coo_matrix(adj)
    adj = adj + adj.T.multiply(adj.T > adj) + sp.eye(nodes_num)
    nor_adj = normalize_adj(adj)
    nor_adj = np.array(nor_adj.todense())
    return nor_adj, features, labels_onehot


def get_mask(total, begin, end):
    """Generate mask."""
    mask = np.zeros([total]).astype(np.float32)
    mask[begin:end] = 1
    return mask
