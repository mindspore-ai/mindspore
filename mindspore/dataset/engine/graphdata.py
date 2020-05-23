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
graphdata.py supports loading graph dataset for GNN network training,
and provides operations related to graph data.
"""
import numpy as np
from mindspore._c_dataengine import Graph
from mindspore._c_dataengine import Tensor

from .validators import check_gnn_graphdata, check_gnn_get_all_nodes, check_gnn_get_all_neighbors, \
    check_gnn_get_node_feature


class GraphData:
    """
    Reads the graph dataset used for GNN training from the shared file and database.

    Args:
        dataset_file (str): One of file names in dataset.
        num_parallel_workers (int, optional): Number of workers to process the Dataset in parallel
            (default=None).
    """

    @check_gnn_graphdata
    def __init__(self, dataset_file, num_parallel_workers=None):
        self._dataset_file = dataset_file
        if num_parallel_workers is None:
            num_parallel_workers = 1
        self._graph = Graph(dataset_file, num_parallel_workers)

    @check_gnn_get_all_nodes
    def get_all_nodes(self, node_type):
        """
        Get all nodes in the graph.

        Args:
            node_type (int): Specify the type of node.

        Returns:
            numpy.ndarray: array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)

        Raises:
            TypeError: If `node_type` is not integer.
        """
        return self._graph.get_nodes(node_type, -1).as_array()

    @check_gnn_get_all_neighbors
    def get_all_neighbors(self, node_list, neighbor_type):
        """
        Get `neighbor_type` neighbors of the nodes in `node_list`.

        Args:
            node_list (list or numpy.ndarray): The given list of nodes.
            neighbor_type (int): Specify the type of neighbor.

        Returns:
            numpy.ndarray: array of nodes.

        Examples:
            >>> import mindspore.dataset as ds
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> neighbors = data_graph.get_all_neighbors(nodes[0], 0)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_type` is not integer.
        """
        return self._graph.get_all_neighbors(node_list, neighbor_type).as_array()

    @check_gnn_get_node_feature
    def get_node_feature(self, node_list, feature_types):
        """
        Get `feature_types` feature of the nodes in `node_list`.

        Args:
            node_list (list or numpy.ndarray): The given list of nodes.
            feature_types (list or ndarray): The given list of feature types.

        Returns:
            numpy.ndarray: array of features.

        Examples:
            >>> import mindspore.dataset as ds
            >>> data_graph = ds.GraphData('dataset_file', 2)
            >>> nodes = data_graph.get_all_nodes(0)
            >>> features = data_graph.get_node_feature(nodes[0], [1])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if isinstance(node_list, list):
            node_list = np.array(node_list, dtype=np.int32)
        return [t.as_array() for t in self._graph.get_node_feature(Tensor(node_list), feature_types)]
