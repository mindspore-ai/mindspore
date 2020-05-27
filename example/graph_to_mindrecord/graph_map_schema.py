# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License")
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
Graph data convert tool for MindRecord.
"""
import numpy as np

__all__ = ['GraphMapSchema']


class GraphMapSchema:
    """
    Class is for transformation from graph data to MindRecord.
    """

    def __init__(self):
        """
        init
        """
        self.num_node_features = 0
        self.num_edge_features = 0
        self.union_schema_in_mindrecord = {
            "first_id": {"type": "int64"},
            "second_id": {"type": "int64"},
            "third_id": {"type": "int64"},
            "type": {"type": "int32"},
            "attribute": {"type": "string"},  # 'n' for ndoe, 'e' for edge
            "node_feature_index": {"type": "int32", "shape": [-1]},
            "edge_feature_index": {"type": "int32", "shape": [-1]}
        }

    def get_schema(self):
        """
        Get schema
        """
        return self.union_schema_in_mindrecord

    def set_node_feature_profile(self, num_features, features_data_type, features_shape):
        """
        Set node features profile
        """
        if num_features != len(features_data_type) or num_features != len(features_shape):
            raise ValueError("Node feature profile is not match.")

        self.num_node_features = num_features
        for i in range(num_features):
            k = i + 1
            field_key = 'node_feature_' + str(k)
            field_value = {"type": features_data_type[i], "shape": features_shape[i]}
            self.union_schema_in_mindrecord[field_key] = field_value

    def set_edge_feature_profile(self, num_features, features_data_type, features_shape):
        """
        Set edge features profile
        """
        if num_features != len(features_data_type) or num_features != len(features_shape):
            raise ValueError("Edge feature profile is not match.")

        self.num_edge_features = num_features
        for i in range(num_features):
            k = i + 1
            field_key = 'edge_feature_' + str(k)
            field_value = {"type": features_data_type[i], "shape": features_shape[i]}
            self.union_schema_in_mindrecord[field_key] = field_value

    def transform_node(self, node):
        """
        Executes transformation from node data to union format.
        Args:
            node(schema): node's data
        Returns:
            graph data with union schema
        """
        node_graph = {"first_id": node["id"], "second_id": 0, "third_id": 0, "attribute": 'n', "type": node["type"],
                      "node_feature_index": []}
        for i in range(self.num_node_features):
            k = i + 1
            node_field_key = 'feature_' + str(k)
            graph_field_key = 'node_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            if node_field_key in node:
                node_graph["node_feature_index"].append(k)
                node_graph[graph_field_key] = np.reshape(np.array(node[node_field_key], dtype=graph_field_type), [-1])
            else:
                node_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])

        if node_graph["node_feature_index"]:
            node_graph["node_feature_index"] = np.array(node_graph["node_feature_index"], dtype="int32")
        else:
            node_graph["node_feature_index"] = np.array([-1], dtype="int32")

        node_graph["edge_feature_index"] = np.array([-1], dtype="int32")
        for i in range(self.num_edge_features):
            k = i + 1
            graph_field_key = 'edge_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            node_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])
        return node_graph

    def transform_edge(self, edge):
        """
        Executes transformation from edge data to union format.
        Args:
            edge(schema): edge's data
        Returns:
            graph data with union schema
        """
        edge_graph = {"first_id": edge["id"], "second_id": edge["src_id"], "third_id": edge["dst_id"], "attribute": 'e',
                      "type": edge["type"], "edge_feature_index": []}

        for i in range(self.num_edge_features):
            k = i + 1
            edge_field_key = 'feature_' + str(k)
            graph_field_key = 'edge_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            if edge_field_key in edge:
                edge_graph["edge_feature_index"].append(k)
                edge_graph[graph_field_key] = np.reshape(np.array(edge[edge_field_key], dtype=graph_field_type), [-1])
            else:
                edge_graph[graph_field_key] = np.reshape(np.array([0], dtype=graph_field_type), [-1])

        if edge_graph["edge_feature_index"]:
            edge_graph["edge_feature_index"] = np.array(edge_graph["edge_feature_index"], dtype="int32")
        else:
            edge_graph["edge_feature_index"] = np.array([-1], dtype="int32")

        edge_graph["node_feature_index"] = np.array([-1], dtype="int32")
        for i in range(self.num_node_features):
            k = i + 1
            graph_field_key = 'node_feature_' + str(k)
            graph_field_type = self.union_schema_in_mindrecord[graph_field_key]["type"]
            edge_graph[graph_field_key] = np.array([0], dtype=graph_field_type)
        return edge_graph
