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
"""Preprocess data obtained for training"""
import numpy as np
import mindspore.dataset as ds


def adj_to_bias(adj):
    """Add self loop to adj and make sure only one hop neighbors are engaged in computing"""
    num_graphs = adj.shape[0]
    adj_temp = np.empty(adj.shape)
    for i in range(num_graphs):
        adj_temp[i] = adj[i] + np.eye(adj.shape[1])
    return -1e9 * (1.0 - adj_temp)


def get_biases_features_labels(data_dir):
    """Get biases, features, labels from Dataset"""
    g = ds.GraphData(data_dir)
    nodes = g.get_all_nodes(0)
    nodes_list = nodes.tolist()
    row_tensor = g.get_node_feature(nodes_list, [1, 2])
    features = row_tensor[0]
    features = features[np.newaxis]

    labels = row_tensor[1]

    nodes_num = labels.shape[0]
    class_num = labels.max() + 1
    labels_onehot = np.eye(nodes_num, class_num)[labels].astype(np.float32)

    neighbor = g.get_all_neighbors(nodes_list, 0)
    node_map = {node_id: index for index, node_id in enumerate(nodes_list)}
    adj = np.zeros([nodes_num, nodes_num], dtype=np.float32)
    for index, value in np.ndenumerate(neighbor):
        if value >= 0 and index[1] > 0:
            adj[node_map[neighbor[index[0], 0]], node_map[value]] = 1
    adj = adj[np.newaxis]
    biases = adj_to_bias(adj)

    return biases, features, labels_onehot


def get_mask(total, begin, end):
    """Generate mask according to begin and end position"""
    mask = np.zeros([total]).astype(np.float32)
    mask[begin:end] = 1
    return np.array(mask, dtype=np.bool)


def load_and_process(data_dir, train_node_num, eval_node_num, test_node_num):
    """Load cora dataset and preprocessing"""
    biases, feature, label = get_biases_features_labels(data_dir)
    # split training, validation and testing set
    nodes_num = label.shape[0]
    train_mask = get_mask(nodes_num, 0, train_node_num)
    eval_mask = get_mask(nodes_num, train_node_num, train_node_num + eval_node_num)
    test_mask = get_mask(nodes_num, nodes_num - test_node_num, nodes_num)

    y_train = np.zeros(label.shape)
    y_val = np.zeros(label.shape)
    y_test = np.zeros(label.shape)

    y_train[train_mask, :] = label[train_mask, :]
    y_val[eval_mask, :] = label[eval_mask, :]
    y_test[test_mask, :] = label[test_mask, :]

    y_train = y_train[np.newaxis]
    y_val = y_val[np.newaxis]
    y_test = y_test[np.newaxis]
    train_mask = train_mask[np.newaxis]
    eval_mask = eval_mask[np.newaxis]
    test_mask = test_mask[np.newaxis]

    return feature, biases, y_train, train_mask, y_val, eval_mask, y_test, test_mask
