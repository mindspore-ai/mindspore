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
import csv
import os

import numpy as np
import scipy.sparse as sp

# parse args from command line parameter 'graph_api_args'
#     args delimiter is ':'
args = os.environ['graph_api_args'].split(':')
CITESEER_CONTENT_FILE = args[0]
CITESEER_CITES_FILE = args[1]
CITESEER_MINDRECRD_LABEL_FILE = CITESEER_CONTENT_FILE + "_label_mindrecord"
CITESEER_MINDRECRD_ID_MAP_FILE = CITESEER_CONTENT_FILE + "_id_mindrecord"

node_id_map = {}

# profile:  (num_features, feature_data_types, feature_shapes)
node_profile = (2, ["float32", "int64"], [[-1], [-1]])
edge_profile = (0, [], [])


def _normalize_citeseer_features(features):
    features = np.array(features)
    row_sum = np.array(features.sum(1))
    r_inv = np.power(row_sum * 1.0, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features


def yield_nodes(task_id=0):
    """
    Generate node data

    Yields:
        data (dict): data row which is dict.
    """
    print("Node task is {}".format(task_id))
    label_types = {}
    label_size = 0
    node_num = 0
    with open(CITESEER_CONTENT_FILE) as content_file:
        content_reader = csv.reader(content_file, delimiter='\t')
        line_count = 0
        for row in content_reader:
            if not row[-1] in label_types:
                label_types[row[-1]] = label_size
                label_size += 1
            if not row[0] in node_id_map:
                node_id_map[row[0]] = node_num
                node_num += 1
            raw_features = [[int(x) for x in row[1:-1]]]
            node = {'id': node_id_map[row[0]], 'type': 0, 'feature_1': _normalize_citeseer_features(raw_features),
                    'feature_2': [label_types[row[-1]]]}
            yield node
            line_count += 1
    print('Processed {} lines for nodes.'.format(line_count))
    # print('label types {}.'.format(label_types))
    with open(CITESEER_MINDRECRD_LABEL_FILE, 'w') as f:
        for k in label_types:
            print(k + ',' + str(label_types[k]), file=f)


def yield_edges(task_id=0):
    """
    Generate edge data

    Yields:
        data (dict): data row which is dict.
    """
    print("Edge task is {}".format(task_id))
    # print(map_string_int)
    with open(CITESEER_CITES_FILE) as cites_file:
        cites_reader = csv.reader(cites_file, delimiter='\t')
        line_count = 0
        for row in cites_reader:
            if not row[0] in node_id_map:
                print('Source node {} does not exist.'.format(row[0]))
                continue
            if not row[1] in node_id_map:
                print('Destination node {} does not exist.'.format(row[1]))
                continue
            line_count += 1
            edge = {'id': line_count,
                    'src_id': node_id_map[row[0]], 'dst_id': node_id_map[row[1]], 'type': 0}
            yield edge

        with open(CITESEER_MINDRECRD_ID_MAP_FILE, 'w') as f:
            for k in node_id_map:
                print(k + ',' + str(node_id_map[k]), file=f)
        print('Processed {} lines for edges.'.format(line_count))
