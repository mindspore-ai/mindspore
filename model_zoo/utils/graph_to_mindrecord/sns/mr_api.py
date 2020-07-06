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
social_data = [[348, 350], [348, 327], [348, 329], [348, 331], [348, 335],
               [348, 336], [348, 337], [348, 338], [348, 340], [348, 341],
               [348, 342], [348, 343], [348, 344], [348, 345], [348, 346],
               [348, 347], [347, 351], [347, 327], [347, 329], [347, 331],
               [347, 335], [347, 341], [347, 345], [347, 346], [346, 335],
               [346, 340], [346, 339], [346, 349], [346, 353], [346, 354],
               [346, 341], [346, 345], [345, 335], [345, 336], [345, 341],
               [344, 338], [344, 342], [343, 332], [343, 338], [343, 342],
               [342, 332], [340, 349], [334, 349], [333, 349], [330, 349],
               [328, 349], [359, 349], [358, 352], [358, 349], [358, 354],
               [358, 356], [357, 350], [357, 354], [357, 356], [356, 350],
               [355, 352], [353, 350], [352, 349], [351, 349], [350, 349]]

# profile:  (num_features, feature_data_types, feature_shapes)
node_profile = (0, [], [])
edge_profile = (0, [], [])


def yield_nodes(task_id=0):
    """
    Generate node data

    Yields:
        data (dict): data row which is dict.
    """
    print("Node task is {}".format(task_id))
    node_list = []
    for edge in social_data:
        src, dst = edge
        if src not in node_list:
            node_list.append(src)
        if dst not in node_list:
            node_list.append(dst)
    node_list.sort()
    print(node_list)
    for node_id in node_list:
        node = {'id': node_id, 'type': 1}
        yield node


def yield_edges(task_id=0):
    """
    Generate edge data

    Yields:
        data (dict): data row which is dict.
    """
    print("Edge task is {}".format(task_id))
    line_count = 0
    for undirected_edge in social_data:
        line_count += 1
        edge = {
            'id': line_count,
            'src_id': undirected_edge[0],
            'dst_id': undirected_edge[1],
            'type': 1}
        yield edge
        line_count += 1
        edge = {
            'id': line_count,
            'src_id': undirected_edge[1],
            'dst_id': undirected_edge[0],
            'type': 1}
        yield edge
