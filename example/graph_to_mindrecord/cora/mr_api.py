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
# ==============================================================================
"""
User-defined API for MindRecord GNN writer.
"""
import os

import pickle as pkl
import numpy as np
import scipy.sparse as sp

# parse args from command line parameter 'graph_api_args'
#     args delimiter is ':'
args = os.environ['graph_api_args'].split(':')
CORA_PATH = args[0]
dataset_str = 'cora'

# profile:  (num_features, feature_data_types, feature_shapes)
node_profile = (2, ["float32", "int32"], [[-1], [-1]])
edge_profile = (0, [], [])


def _normalize_cora_features(features):
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum * 1.0, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def _parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def yield_nodes(task_id=0):
    """
    Generate node data

    Yields:
        data (dict): data row which is dict.
    """
    print("Node task is {}".format(task_id))

    names = ['tx', 'ty', 'allx', 'ally']
    objects = []
    for name in names:
        with open("{}/ind.{}.{}".format(CORA_PATH, dataset_str, name), 'rb') as f:
            objects.append(pkl.load(f, encoding='latin1'))
    tx, ty, allx, ally = tuple(objects)
    test_idx_reorder = _parse_index_file(
        "{}/ind.{}.test.index".format(CORA_PATH, dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _normalize_cora_features(features)
    features = features.A

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    line_count = 0
    for i, label in enumerate(labels):
        node = {'id': i, 'type': 0, 'feature_1': features[i].tolist(),
                'feature_2': label.tolist().index(1)}
        line_count += 1
        yield node
    print('Processed {} lines for nodes.'.format(line_count))


def yield_edges(task_id=0):
    """
    Generate edge data

    Yields:
        data (dict): data row which is dict.
    """
    print("Edge task is {}".format(task_id))
    with open("{}/ind.{}.graph".format(CORA_PATH, dataset_str), 'rb') as f:
        graph = pkl.load(f, encoding='latin1')
        line_count = 0
        for i in graph:
            for dst_id in graph[i]:
                edge = {'id': line_count,
                        'src_id': i, 'dst_id': dst_id, 'type': 0}
                line_count += 1
                yield edge
        print('Processed {} lines for edges.'.format(line_count))
