# Copyright 2020-2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0(the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http:  // www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == == ==
"""
graphdata.py supports loading graph dataset for GNN network training,
and provides operations related to graph data.
"""
import atexit
import os
import random
import time
from enum import IntEnum
import numpy as np
from mindspore._c_dataengine import GraphDataClient
from mindspore._c_dataengine import GraphDataServer
from mindspore._c_dataengine import Tensor
from mindspore._c_dataengine import SamplingStrategy as Sampling
from mindspore._c_dataengine import OutputFormat as Format

from .validators import check_gnn_graphdata, check_gnn_get_all_nodes, check_gnn_get_all_edges, \
    check_gnn_get_nodes_from_edges, check_gnn_get_edges_from_nodes, check_gnn_get_all_neighbors, \
    check_gnn_get_sampled_neighbors, check_gnn_get_neg_sampled_neighbors, check_gnn_get_node_feature, \
    check_gnn_get_edge_feature, check_gnn_random_walk, check_gnn_graph, check_gnn_get_graph_feature
from ..core.validator_helpers import replace_none
from .datasets_user_defined import GeneratorDataset


class SamplingStrategy(IntEnum):
    """
    Specifies the sampling strategy when execute `get_sampled_neighbors` .

    - RANDOM: Random sampling with replacement.
    - EDGE_WEIGHT: Sampling with edge weight as probability.
    """
    RANDOM = 0
    EDGE_WEIGHT = 1


DE_C_INTER_SAMPLING_STRATEGY = {
    SamplingStrategy.RANDOM: Sampling.DE_SAMPLING_RANDOM,
    SamplingStrategy.EDGE_WEIGHT: Sampling.DE_SAMPLING_EDGE_WEIGHT,
}


class OutputFormat(IntEnum):
    """
    Specifies the output storage format when execute `get_all_neighbors` .

    - NORMAL: Normal format.
    - COO: COO format.
    - CSR: CSR format.
    """
    NORMAL = 0
    COO = 1
    CSR = 2


DE_C_INTER_OUTPUT_FORMAT = {
    OutputFormat.NORMAL: Format.DE_FORMAT_NORMAL,
    OutputFormat.COO: Format.DE_FORMAT_COO,
    OutputFormat.CSR: Format.DE_FORMAT_CSR,
}


class GraphData:
    """
    Reads the graph dataset used for GNN training from the shared file and database.
    Support reading graph datasets like Cora, Citeseer and PubMed.

    About how to load raw graph dataset into MindSpore please
    refer to `Loading Graph Dataset <https://www.mindspore.cn/tutorials/en/
    master/advanced/dataset/augment_graph_data.html>`_ .

    Args:
        dataset_file (str): One of file names in the dataset.
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel.
            Default: None.
        working_mode (str, optional): Set working mode, now supports 'local'/'client'/'server'. Default: 'local'.

            - 'local', used in non-distributed training scenarios.

            - 'client', used in distributed training scenarios. The client does not load data,
              but obtains data from the server.

            - 'server', used in distributed training scenarios. The server loads the data
              and is available to the client.

        hostname (str, optional): Hostname of the graph data server. This parameter is only valid when
            `working_mode` is set to 'client' or 'server'. Default: '127.0.0.1'.
        port (int, optional): Port of the graph data server. The range is 1024-65535. This parameter is
            only valid when `working_mode` is set to 'client' or 'server'. Default: 50051.
        num_client (int, optional): Maximum number of clients expected to connect to the server. The server will
            allocate resources according to this parameter. This parameter is only valid when `working_mode`
            is set to 'server'. Default: 1.
        auto_shutdown (bool, optional): Valid when `working_mode` is set to 'server',
            when the number of connected clients reaches `num_client` and no client is being connected,
            the server automatically exits. Default: True.

    Raises:
        ValueError: If `dataset_file` does not exist or permission denied.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `working_mode` is not 'local', 'client' or 'server'.
        TypeError: If `hostname` is illegal.
        ValueError: If `port` is not in range [1024, 65535].
        ValueError: If `num_client` is not in range [1, 255].

    Supported Platforms:
        ``CPU``

    Examples:
        >>> graph_dataset_dir = "/path/to/graph_dataset_file"
        >>> graph_data = ds.GraphData(dataset_file=graph_dataset_dir, num_parallel_workers=2)
        >>> nodes = graph_data.get_all_nodes(node_type=1)
        >>> features = graph_data.get_node_feature(node_list=nodes, feature_types=[1])
    """

    @check_gnn_graphdata
    def __init__(self, dataset_file, num_parallel_workers=None, working_mode='local', hostname='127.0.0.1', port=50051,
                 num_client=1, auto_shutdown=True):
        self._dataset_file = dataset_file
        self._working_mode = working_mode
        self.data_format = "mindrecord"
        if num_parallel_workers is None:
            num_parallel_workers = 1

        if working_mode in ['local', 'client']:
            self._graph_data = GraphDataClient(self.data_format, dataset_file, num_parallel_workers, working_mode,
                                               hostname, port)
            atexit.register(self._stop)

        if working_mode == 'server':
            self._graph_data = GraphDataServer(
                self.data_format, dataset_file, num_parallel_workers, hostname, port, num_client, auto_shutdown)
            atexit.register(self._stop)
            try:
                while self._graph_data.is_stopped() is not True:
                    time.sleep(1)
            except KeyboardInterrupt:
                raise Exception("Graph data server receives KeyboardInterrupt.")

    @check_gnn_get_all_nodes
    def get_all_nodes(self, node_type):
        """
        Get all nodes in the graph.

        Args:
            node_type (int): Specify the type of node.

        Returns:
            numpy.ndarray, array of nodes.

        Examples:
            >>> nodes = graph_data.get_all_nodes(node_type=1)

        Raises:
            TypeError: If `node_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_all_nodes(node_type).as_array()

    @check_gnn_get_all_edges
    def get_all_edges(self, edge_type):
        """
        Get all edges in the graph.

        Args:
            edge_type (int): Specify the type of edge.

        Returns:
            numpy.ndarray, array of edges.

        Examples:
            >>> edges = graph_data.get_all_edges(edge_type=0)

        Raises:
            TypeError: If `edge_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_all_edges(edge_type).as_array()

    @check_gnn_get_nodes_from_edges
    def get_nodes_from_edges(self, edge_list):
        """
        Get nodes from the edges.

        Args:
            edge_list (Union[list, numpy.ndarray]): The given list of edges.

        Returns:
            numpy.ndarray, array of nodes.

        Examples:
            >>> from mindspore.dataset import GraphData
            >>>
            >>> g = ds.GraphData("/path/to/testdata", 1)
            >>> edges = g.get_all_edges(0)
            >>> nodes = g.get_nodes_from_edges(edges)

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_nodes_from_edges(edge_list).as_array()

    @check_gnn_get_edges_from_nodes
    def get_edges_from_nodes(self, node_list):
        """
        Get edges from the nodes.

        Args:
            node_list (Union[list[tuple], numpy.ndarray]): The given list of pair nodes ID.

        Returns:
            numpy.ndarray, array of edges ID.

        Examples:
            >>> edges = graph_data.get_edges_from_nodes(node_list=[(101, 201), (103, 207)])

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_edges_from_nodes(node_list).as_array()

    @check_gnn_get_all_neighbors
    def get_all_neighbors(self, node_list, neighbor_type, output_format=OutputFormat.NORMAL):
        """
        Get `neighbor_type` neighbors of the nodes in `node_list` .
        We try to use the following example to illustrate the definition of these formats. 1 represents connected
        between two nodes, and 0 represents not connected.

        .. list-table:: Adjacent Matrix
           :widths: 20 20 20 20 20
           :header-rows: 1

           * -
             - 0
             - 1
             - 2
             - 3
           * - 0
             - 0
             - 1
             - 0
             - 0
           * - 1
             - 0
             - 0
             - 1
             - 0
           * - 2
             - 1
             - 0
             - 0
             - 1
           * - 3
             - 1
             - 0
             - 0
             - 0

        .. list-table:: Normal Format
           :widths: 20 20 20 20 20
           :header-rows: 1

           * - src
             - 0
             - 1
             - 2
             - 3
           * - dst_0
             - 1
             - 2
             - 0
             - 1
           * - dst_1
             - -1
             - -1
             - 3
             - -1

        .. list-table:: COO Format
           :widths: 20 20 20 20 20 20
           :header-rows: 1

           * - src
             - 0
             - 1
             - 2
             - 2
             - 3
           * - dst
             - 1
             - 2
             - 0
             - 3
             - 1

        .. list-table:: CSR Format
           :widths: 40 20 20 20 20 20
           :header-rows: 1

           * - offsetTable
             - 0
             - 1
             - 2
             - 4
             -
           * - dstTable
             - 1
             - 2
             - 0
             - 3
             - 1

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_type (int): Specify the type of neighbor node.
            output_format (OutputFormat, optional): Output storage format. Default: OutputFormat.NORMAL.
                It can be any of [OutputFormat.NORMAL, OutputFormat.COO, OutputFormat.CSR].

        Returns:
            For NORMAL format or COO format
            numpy.ndarray which represents the array of neighbors will return.
            As if CSR format is specified, two numpy.ndarrays will return.
            The first one is offset table, the second one is neighbors

        Examples:
            >>> from mindspore.dataset.engine import OutputFormat
            >>> nodes = graph_data.get_all_nodes(node_type=1)
            >>> neighbors = graph_data.get_all_neighbors(node_list=nodes, neighbor_type=2)
            >>> neighbors_coo = graph_data.get_all_neighbors(node_list=nodes, neighbor_type=2,
            ...                                              output_format=OutputFormat.COO)
            >>> offset_table, neighbors_csr = graph_data.get_all_neighbors(node_list=nodes, neighbor_type=2,
            ...                                                            output_format=OutputFormat.CSR)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        result_list = self._graph_data.get_all_neighbors(node_list, neighbor_type,
                                                         DE_C_INTER_OUTPUT_FORMAT[output_format]).as_array()
        if output_format == OutputFormat.CSR:
            offset_table = result_list[:len(node_list)]
            neighbor_table = result_list[len(node_list):]
            return offset_table, neighbor_table
        return result_list

    @check_gnn_get_sampled_neighbors
    def get_sampled_neighbors(self, node_list, neighbor_nums, neighbor_types, strategy=SamplingStrategy.RANDOM):
        """
        Get sampled neighbor information.

        The api supports multi-hop neighbor sampling. That is, the previous sampling result is used as the input of
        next-hop sampling. A maximum of 6-hop are allowed.

        The sampling result is tiled into a list in the format of [input node, 1-hop sampling result,
        2-hop sampling result ...].

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_nums (Union[list, numpy.ndarray]): Number of neighbors sampled per hop.
            neighbor_types (Union[list, numpy.ndarray]): Neighbor type sampled per hop, type of each element in
                neighbor_types should be int.
            strategy (SamplingStrategy, optional): Sampling strategy. Default: SamplingStrategy.RANDOM.
                It can be any of [SamplingStrategy.RANDOM, SamplingStrategy.EDGE_WEIGHT].

                - SamplingStrategy.RANDOM, random sampling with replacement.
                - SamplingStrategy.EDGE_WEIGHT, sampling with edge weight as probability.

        Returns:
            numpy.ndarray, array of neighbors.

        Examples:
            >>> nodes = graph_data.get_all_nodes(node_type=1)
            >>> neighbors = graph_data.get_sampled_neighbors(node_list=nodes, neighbor_nums=[2, 2],
            ...                                              neighbor_types=[2, 1])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_nums` is not list or ndarray.
            TypeError: If `neighbor_types` is not list or ndarray.
        """
        if not isinstance(strategy, SamplingStrategy):
            raise TypeError("Wrong input type for strategy, should be enum of 'SamplingStrategy'.")
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_sampled_neighbors(
            node_list, neighbor_nums, neighbor_types, DE_C_INTER_SAMPLING_STRATEGY.get(strategy)).as_array()

    @check_gnn_get_neg_sampled_neighbors
    def get_neg_sampled_neighbors(self, node_list, neg_neighbor_num, neg_neighbor_type):
        """
        Get `neg_neighbor_type` negative sampled neighbors of the nodes in `node_list` .

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neg_neighbor_num (int): Number of neighbors sampled.
            neg_neighbor_type (int): Specify the type of negative neighbor.

        Returns:
            numpy.ndarray, array of neighbors.

        Examples:
            >>> nodes = graph_data.get_all_nodes(node_type=1)
            >>> neg_neighbors = graph_data.get_neg_sampled_neighbors(node_list=nodes, neg_neighbor_num=5,
            ...                                                      neg_neighbor_type=2)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neg_neighbor_num` is not integer.
            TypeError: If `neg_neighbor_type` is not integer.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.get_neg_sampled_neighbors(
            node_list, neg_neighbor_num, neg_neighbor_type).as_array()

    @check_gnn_get_node_feature
    def get_node_feature(self, node_list, feature_types):
        """
        Get `feature_types` feature of the nodes in `node_list` .

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types.

        Returns:
            numpy.ndarray, array of features.

        Examples:
            >>> nodes = graph_data.get_all_nodes(node_type=1)
            >>> features = graph_data.get_node_feature(node_list=nodes, feature_types=[2, 3])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        if isinstance(node_list, list):
            node_list = np.array(node_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_node_feature(
                Tensor(node_list),
                feature_types)]

    @check_gnn_get_edge_feature
    def get_edge_feature(self, edge_list, feature_types):
        """
        Get `feature_types` feature of the edges in `edge_list` .

        Args:
            edge_list (Union[list, numpy.ndarray]): The given list of edges.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types.

        Returns:
            numpy.ndarray, array of features.

        Examples:
            >>> edges = graph_data.get_all_edges(edge_type=0)
            >>> features = graph_data.get_edge_feature(edge_list=edges, feature_types=[1])

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        if isinstance(edge_list, list):
            edge_list = np.array(edge_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_edge_feature(
                Tensor(edge_list),
                feature_types)]

    def graph_info(self):
        """
        Get the meta information of the graph, including the number of nodes, the type of nodes,
        the feature information of nodes, the number of edges, the type of edges, and the feature information of edges.

        Returns:
            dict, meta information of the graph. The key is node_type, edge_type, node_num, edge_num,
            node_feature_type and edge_feature_type.

        Examples:
        >>> from mindspore.dataset import GraphData
        >>>
        >>> g = ds.GraphData("/path/to/testdata", 2)
        >>> graph_info = g.graph_info()
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.graph_info()

    @check_gnn_random_walk
    def random_walk(self, target_nodes, meta_path, step_home_param=1.0, step_away_param=1.0, default_node=-1):
        """
        Random walk in nodes.

        Args:
            target_nodes (list[int]): Start node list in random walk
            meta_path (list[int]): node type for each walk step
            step_home_param (float, optional): return hyper parameter in node2vec algorithm. Default: 1.0.
            step_away_param (float, optional): in out hyper parameter in node2vec algorithm. Default: 1.0.
            default_node (int, optional): default node if no more neighbors found. Default: -1.
                A default value of -1 indicates that no node is given.

        Returns:
            numpy.ndarray, array of nodes.

        Examples:
            >>> nodes = graph_data.get_all_nodes(node_type=1)
            >>> walks = graph_data.random_walk(target_nodes=nodes, meta_path=[2, 1, 2])

        Raises:
            TypeError: If `target_nodes` is not list or ndarray.
            TypeError: If `meta_path` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        return self._graph_data.random_walk(target_nodes, meta_path, step_home_param, step_away_param,
                                            default_node).as_array()

    def _stop(self):
        """Stop GraphDataClient or GraphDataServer."""
        self._graph_data.stop()


class Graph(GraphData):
    """
    A graph object for storing Graph structure and feature data, and provide capabilities such as graph sampling.

    This class supports init graph With input numpy array data, which represent node, edge and its features.
    If working mode is `local` , there is no need to specify input arguments like `working_mode` , `hostname` , `port` ,
    `num_client` , `auto_shutdown` .

    Args:
        edges(Union[list, numpy.ndarray]): edges of graph in COO format with shape [2, num_edges].
        node_feat(dict, optional): feature of nodes, input data format should be dict, key is feature type, which is
            represented with string like 'weight' etc, value should be numpy.array with shape
            [num_nodes, num_node_features].
        edge_feat(dict, optional): feature of edges, input data format should be dict, key is feature type, which is
            represented with string like 'weight' etc, value should be numpy.array with shape
            [num_edges, num_edge_features].
        graph_feat(dict, optional): additional feature, which can not be assigned to node_feat or edge_feat, input data
            format should be dict, key is feature type, which is represented with string, value should be numpy.array,
            its shape is not restricted.
        node_type(Union[list, numpy.ndarray], optional): type of nodes, each element should be string which represent
            type of corresponding node. If not provided, default type for each node is "0".
        edge_type(Union[list, numpy.ndarray], optional): type of edges, each element should be string which represent
            type of corresponding edge. If not provided, default type for each edge is "0".
        num_parallel_workers (int, optional): Number of workers to process the dataset in parallel. Default: None.
        working_mode (str, optional): Set working mode, now supports 'local'/'client'/'server'. Default: 'local'.

            - 'local', used in non-distributed training scenarios.

            - 'client', used in distributed training scenarios. The client does not load data,
              but obtains data from the server.

            - 'server', used in distributed training scenarios. The server loads the data
              and is available to the client.

        hostname (str, optional): Hostname of the graph data server. This parameter is only valid when
            `working_mode` is set to 'client' or 'server'. Default: '127.0.0.1'.
        port (int, optional): Port of the graph data server. The range is 1024-65535. This parameter is
            only valid when `working_mode` is set to 'client' or 'server'. Default: 50051.
        num_client (int, optional): Maximum number of clients expected to connect to the server. The server will
            allocate resources according to this parameter. This parameter is only valid when `working_mode`
            is set to 'server'. Default: 1.
        auto_shutdown (bool, optional): Valid when `working_mode` is set to 'server',
            when the number of connected clients reaches `num_client` and no client is being connected,
            the server automatically exits. Default: True.

    Raises:
        TypeError: If `edges` not list or NumPy array.
        TypeError: If `node_feat` provided but not dict, or key in dict is not string type, or value in dict not NumPy
            array.
        TypeError: If `edge_feat` provided but not dict, or key in dict is not string type, or value in dict not NumPy
            array.
        TypeError: If `graph_feat` provided but not dict, or key in dict is not string type, or value in dict not NumPy
            array.
        TypeError: If `node_type` provided but its type not list or NumPy array.
        TypeError: If `edge_type` provided but its type not list or NumPy array.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.
        ValueError: If `working_mode` is not 'local', 'client' or 'server'.
        TypeError: If `hostname` is illegal.
        ValueError: If `port` is not in range [1024, 65535].
        ValueError: If `num_client` is not in range [1, 255].

    Examples:
        >>> import numpy as np
        >>> from mindspore.dataset import Graph
        >>>
        >>> # 1) Only provide edges for creating graph, as this is the only required input parameter
        >>> edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
        >>> graph = Graph(edges)
        >>> graph_info = graph.graph_info()
        >>>
        >>> # 2) Setting node_feat and edge_feat for corresponding node and edge
        >>> #    first dimension of feature shape should be corresponding node num or edge num.
        >>> edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
        >>> node_feat = {"node_feature_1": np.array([[0], [1], [2]], dtype=np.int32)}
        >>> edge_feat = {"edge_feature_1": np.array([[1, 2], [3, 4]], dtype=np.int32)}
        >>> graph = Graph(edges, node_feat, edge_feat)
        >>>
        >>> # 3) Setting graph feature for graph, there is no shape limit for graph feature
        >>> edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
        >>> graph_feature = {"graph_feature_1": np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)}
        >>> graph = Graph(edges, graph_feat=graph_feature)
    """

    @check_gnn_graph
    def __init__(self, edges, node_feat=None, edge_feat=None, graph_feat=None, node_type=None, edge_type=None,
                 num_parallel_workers=None, working_mode='local', hostname='127.0.0.1', port=50051, num_client=1,
                 auto_shutdown=True):
        node_feat = replace_none(node_feat, {})
        edge_feat = replace_none(edge_feat, {})
        graph_feat = replace_none(graph_feat, {})
        edges = np.asarray(edges, dtype=np.int32)
        # infer num_nodes
        num_nodes = np.max(edges) + 1

        if node_type is not None:
            node_type = np.asarray(node_type)
            if len(node_type.shape) != 1 or node_type.shape[0] != num_nodes:
                raise ValueError(
                    "Input 'node_type' should be of 1 dimension, and its length should be {}, but got {}.".format(
                        num_nodes, len(node_type)))
        else:
            node_type = np.array(["0"] * num_nodes)

        edge_type = replace_none(edge_type, np.array(["0"] * edges.shape[1]))
        edge_type = np.asarray(edge_type)

        self._working_mode = working_mode
        self.data_format = "array"
        self.node_type_mapping, self.edge_type_mapping = dict(), dict()
        self.node_feature_type_mapping, self.edge_feature_type_mapping = dict(), dict()
        self.graph_feature_type_mapping = dict()
        self.invert_node_type_mapping, self.invert_edge_type_mapping = dict(), dict()
        self.invert_node_feature_type_mapping, self.invert_edge_feature_type_mapping = dict(), dict()
        self.invert_graph_feature_type_mapping = dict()

        node_feat, edge_feat, graph_feat, node_type, edge_type = \
            self._replace_string(node_feat, edge_feat, graph_feat, node_type, edge_type)

        if num_parallel_workers is None:
            num_parallel_workers = 1

        if working_mode in ['local', 'client']:
            # GraphDataClient should support different init way, as data might be different
            self._graph_data = GraphDataClient(self.data_format, num_nodes, edges, node_feat, edge_feat, graph_feat,
                                               node_type, edge_type, num_parallel_workers, working_mode, hostname,
                                               port)
            atexit.register(self._stop)

        if working_mode == 'server':
            self._graph_data = GraphDataServer(self.data_format, num_nodes, edges, node_feat, edge_feat, graph_feat,
                                               node_type, edge_type, num_parallel_workers, hostname, port, num_client,
                                               auto_shutdown)
            atexit.register(self._stop)
            try:
                while self._graph_data.is_stopped() is not True:
                    time.sleep(1)
            except KeyboardInterrupt:
                raise Exception("Graph data server receives KeyboardInterrupt.")

    @check_gnn_get_all_nodes
    def get_all_nodes(self, node_type):
        """
        Get all nodes in the graph.

        Args:
            node_type (str): Specify the type of node.

        Returns:
            numpy.ndarray, array of nodes.

        Examples:
            >>> nodes = graph.get_all_nodes(node_type="0")

        Raises:
            TypeError: If `node_type` is not string.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")

        if node_type not in self.node_type_mapping:
            raise ValueError("Given node type {} is not exist in graph, existed is: {}."
                             .format(node_type, list(self.node_type_mapping.keys())))
        node_int_type = self.node_type_mapping[node_type]
        return self._graph_data.get_all_nodes(node_int_type).as_array()

    @check_gnn_get_all_edges
    def get_all_edges(self, edge_type):
        """
        Get all edges in the graph.

        Args:
            edge_type (str): Specify the type of edge, default edge_type is "0" when init graph without specify
                edge_type.

        Returns:
            numpy.ndarray, array of edges.

        Examples:
            >>> edges = graph.get_all_edges(edge_type="0")

        Raises:
            TypeError: If `edge_type` is not string.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")

        if edge_type not in self.edge_type_mapping:
            raise ValueError("Given node type {} is not exist in graph, existed is: {}."
                             .format(edge_type, list(self.edge_type_mapping.keys())))
        edge_int_type = self.node_type_mapping[edge_type]
        return self._graph_data.get_all_edges(edge_int_type).as_array()

    @check_gnn_get_all_neighbors
    def get_all_neighbors(self, node_list, neighbor_type, output_format=OutputFormat.NORMAL):
        """
        Get `neighbor_type` neighbors of the nodes in `node_list` .
        We try to use the following example to illustrate the definition of these formats. 1 represents connected
        between two nodes, and 0 represents not connected.

        .. list-table:: Adjacent Matrix
           :widths: 20 20 20 20 20
           :header-rows: 1

           * -
             - 0
             - 1
             - 2
             - 3
           * - 0
             - 0
             - 1
             - 0
             - 0
           * - 1
             - 0
             - 0
             - 1
             - 0
           * - 2
             - 1
             - 0
             - 0
             - 1
           * - 3
             - 1
             - 0
             - 0
             - 0

        .. list-table:: Normal Format
           :widths: 20 20 20 20 20
           :header-rows: 1

           * - src
             - 0
             - 1
             - 2
             - 3
           * - dst_0
             - 1
             - 2
             - 0
             - 1
           * - dst_1
             - -1
             - -1
             - 3
             - -1

        .. list-table:: COO Format
           :widths: 20 20 20 20 20 20
           :header-rows: 1

           * - src
             - 0
             - 1
             - 2
             - 2
             - 3
           * - dst
             - 1
             - 2
             - 0
             - 3
             - 1

        .. list-table:: CSR Format
           :widths: 40 20 20 20 20 20
           :header-rows: 1

           * - offsetTable
             - 0
             - 1
             - 2
             - 4
             -
           * - dstTable
             - 1
             - 2
             - 0
             - 3
             - 1

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_type (str): Specify the type of neighbor node.
            output_format (OutputFormat, optional): Output storage format. Default: OutputFormat.NORMAL.
                It can be any of [OutputFormat.NORMAL, OutputFormat.COO, OutputFormat.CSR].

        Returns:
            For NORMAL format or COO format
            numpy.ndarray which represents the array of neighbors will return.
            As if CSR format is specified, two numpy.ndarrays will return.
            The first one is offset table, the second one is neighbors

        Examples:
            >>> from mindspore.dataset.engine import OutputFormat
            >>> nodes = graph.get_all_nodes(node_type="0")
            >>> neighbors = graph.get_all_neighbors(node_list=nodes, neighbor_type="0")
            >>> neighbors_coo = graph.get_all_neighbors(node_list=nodes, neighbor_type="0",
            ...                                         output_format=OutputFormat.COO)
            >>> offset_table, neighbors_csr = graph.get_all_neighbors(node_list=nodes, neighbor_type="0",
            ...                                                       output_format=OutputFormat.CSR)

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_type` is not string.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        if neighbor_type not in self.node_type_mapping:
            raise ValueError("Given neighbor node type {} is not exist in graph, existed is: {}."
                             .format(neighbor_type, list(self.node_type_mapping.keys())))
        neighbor_int_type = self.node_type_mapping[neighbor_type]
        result_list = self._graph_data.get_all_neighbors(node_list, neighbor_int_type,
                                                         DE_C_INTER_OUTPUT_FORMAT[output_format]).as_array()
        if output_format == OutputFormat.CSR:
            offset_table = result_list[:len(node_list)]
            neighbor_table = result_list[len(node_list):]
            return offset_table, neighbor_table
        return result_list

    @check_gnn_get_sampled_neighbors
    def get_sampled_neighbors(self, node_list, neighbor_nums, neighbor_types, strategy=SamplingStrategy.RANDOM):
        """
        Get sampled neighbor information.

        The api supports multi-hop neighbor sampling. That is, the previous sampling result is used as the input of
        next-hop sampling. A maximum of 6-hop are allowed.

        The sampling result is tiled into a list in the format of [input node, 1-hop sampling result,
        2-hop sampling result ...].

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neighbor_nums (Union[list, numpy.ndarray]): Number of neighbors sampled per hop.
            neighbor_types (Union[list, numpy.ndarray]): Neighbor type sampled per hop, type of each element in
                neighbor_types should be str.
            strategy (SamplingStrategy, optional): Sampling strategy. Default: SamplingStrategy.RANDOM.
                It can be any of [SamplingStrategy.RANDOM, SamplingStrategy.EDGE_WEIGHT].

                - SamplingStrategy.RANDOM, random sampling with replacement.
                - SamplingStrategy.EDGE_WEIGHT, sampling with edge weight as probability.

        Returns:
            numpy.ndarray, array of neighbors.

        Examples:
            >>> nodes = graph.get_all_nodes(node_type="0")
            >>> neighbors = graph.get_sampled_neighbors(node_list=nodes, neighbor_nums=[2, 2],
            ...                                         neighbor_types=["0", "0"])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neighbor_nums` is not list or ndarray.
            TypeError: If `neighbor_types` is not list or ndarray.
        """
        if not isinstance(strategy, SamplingStrategy):
            raise TypeError("Wrong input type for strategy, should be enum of 'SamplingStrategy'.")
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")

        neighbor_int_types = []
        for neighbor_type in neighbor_types:
            if neighbor_type not in self.node_type_mapping:
                raise ValueError("Given neighbor node type {} is not exist in graph, existed is: {}."
                                 .format(neighbor_type, list(self.node_type_mapping.keys())))
            neighbor_int_types.append(self.node_type_mapping[neighbor_type])
        return self._graph_data.get_sampled_neighbors(
            node_list, neighbor_nums, neighbor_int_types, DE_C_INTER_SAMPLING_STRATEGY.get(strategy)).as_array()

    @check_gnn_get_neg_sampled_neighbors
    def get_neg_sampled_neighbors(self, node_list, neg_neighbor_num, neg_neighbor_type):
        """
        Get `neg_neighbor_type` negative sampled neighbors of the nodes in `node_list` .

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            neg_neighbor_num (int): Number of neighbors sampled.
            neg_neighbor_type (str): Specify the type of negative neighbor.

        Returns:
            numpy.ndarray, array of neighbors.

        Examples:
            >>> nodes = graph.get_all_nodes(node_type="0")
            >>> neg_neighbors = graph.get_neg_sampled_neighbors(node_list=nodes, neg_neighbor_num=3,
            ...                                                 neg_neighbor_type="0")

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `neg_neighbor_num` is not integer.
            TypeError: If `neg_neighbor_type` is not string.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        if neg_neighbor_type not in self.node_type_mapping:
            raise ValueError("Given neighbor node type {} is not exist in graph, existed is: {}"
                             .format(neg_neighbor_type, list(self.node_type_mapping.keys())))
        neg_neighbor_int_type = self.node_type_mapping[neg_neighbor_type]
        return self._graph_data.get_neg_sampled_neighbors(
            node_list, neg_neighbor_num, neg_neighbor_int_type).as_array()

    @check_gnn_get_node_feature
    def get_node_feature(self, node_list, feature_types):
        """
        Get `feature_types` feature of the nodes in `node_list` .

        Args:
            node_list (Union[list, numpy.ndarray]): The given list of nodes.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types, each element should be string.

        Returns:
            numpy.ndarray, array of features.

        Examples:
            >>> nodes = graph.get_all_nodes(node_type="0")
            >>> features = graph.get_node_feature(node_list=nodes, feature_types=["node_feature_1"])

        Raises:
            TypeError: If `node_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")

        feature_int_types = []
        for feature_type in feature_types:
            if feature_type not in self.node_feature_type_mapping:
                raise ValueError("Given node feature type {} is not exist in graph, existed is: {}."
                                 .format(feature_type, list(self.node_feature_type_mapping.keys())))
            feature_int_types.append(self.node_feature_type_mapping[feature_type])
        if isinstance(node_list, list):
            node_list = np.array(node_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_node_feature(
                Tensor(node_list),
                feature_int_types)]

    @check_gnn_get_edge_feature
    def get_edge_feature(self, edge_list, feature_types):
        """
        Get `feature_types` feature of the edges in `edge_list` .

        Args:
            edge_list (Union[list, numpy.ndarray]): The given list of edges.
            feature_types (Union[list, numpy.ndarray]): The given list of feature types, each element should be string.

        Returns:
            numpy.ndarray, array of features.

        Examples:
            >>> edges = graph.get_all_edges(edge_type="0")
            >>> features = graph.get_edge_feature(edge_list=edges, feature_types=["edge_feature_1"])

        Raises:
            TypeError: If `edge_list` is not list or ndarray.
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        feature_int_types = []
        for feature_type in feature_types:
            if feature_type not in self.edge_feature_type_mapping:
                raise ValueError("Given edge feature type {} is not exist in graph, existed is: {}."
                                 .format(feature_type, list(self.edge_feature_type_mapping.keys())))
            feature_int_types.append(self.edge_feature_type_mapping[feature_type])

        if isinstance(edge_list, list):
            edge_list = np.array(edge_list, dtype=np.int32)
        return [
            t.as_array() for t in self._graph_data.get_edge_feature(
                Tensor(edge_list),
                feature_int_types)]

    @check_gnn_get_graph_feature
    def get_graph_feature(self, feature_types):
        """
        Get `feature_types` feature that stored in Graph feature level.

        Args:
            feature_types (Union[list, numpy.ndarray]): The given list of feature types, each element should be string.

        Returns:
            numpy.ndarray, array of features.

        Examples:
            >>> features = graph.get_graph_feature(feature_types=['graph_feature_1'])

        Raises:
            TypeError: If `feature_types` is not list or ndarray.
        """
        if self._working_mode in ['server']:
            raise Exception("This method is not supported when working mode is server.")

        feature_int_types = []
        for feature_type in feature_types:
            if feature_type not in self.graph_feature_type_mapping:
                raise ValueError("Given graph feature type {} is not exist in graph, existed is: {}."
                                 .format(feature_type, list(self.graph_feature_type_mapping.keys())))
            feature_int_types.append(self.graph_feature_type_mapping[feature_type])
        return [t.as_array() for t in self._graph_data.get_graph_feature(feature_int_types)]

    @staticmethod
    def _convert_list(data, mapping):
        """
        Convert list data according to given mapping.
        """
        new_data = []
        for item in data:
            new_data.append(mapping[item])
        return new_data

    @staticmethod
    def _convert_dict(data, mapping):
        """
        Convert dict data according to given mapping.
        """
        new_data = dict()
        for key, value in data.items():
            new_data[mapping[key]] = value
        return new_data

    def graph_info(self):
        """
        Get the meta information of the graph, including the number of nodes, the type of nodes,
        the feature information of nodes, the number of edges, the type of edges, and the feature information of edges.

        Returns:
            dict, meta information of the graph. The key is node_type, edge_type, node_num, edge_num,
            node_feature_type, edge_feature_type and graph_feature_type.
        """
        if self._working_mode == 'server':
            raise Exception("This method is not supported when working mode is server.")
        # do type convert for node_type, edge_type, and other feature_type
        raw_info = self._graph_data.graph_info()
        graph_info = dict()
        graph_info["node_type"] = self._convert_list(raw_info["node_type"], self.invert_node_type_mapping)
        graph_info["edge_type"] = self._convert_list(raw_info["edge_type"], self.invert_edge_type_mapping)
        graph_info["node_feature_type"] = \
            self._convert_list(raw_info["node_feature_type"], self.invert_node_feature_type_mapping)
        graph_info["edge_feature_type"] = \
            self._convert_list(raw_info["edge_feature_type"], self.invert_edge_feature_type_mapping)
        graph_info["graph_feature_type"] = \
            self._convert_list(raw_info["graph_feature_type"], self.invert_graph_feature_type_mapping)
        graph_info["node_num"] = self._convert_dict(raw_info["node_num"], self.invert_node_type_mapping)
        graph_info["edge_num"] = self._convert_dict(raw_info["edge_num"], self.invert_edge_type_mapping)
        return graph_info

    def _replace_string(self, node_feat, edge_feat, graph_feat, node_type, edge_type):
        """
        replace key in node_feat, edge_feat and graph_feat from string into int, and replace value in node_type and
        edge_type from string to int.
        """

        def replace_dict_key(feature):
            index = 0
            new_feature = dict()
            feature_type_mapping = dict()
            for item in feature.items():
                new_feature[index] = item[1]
                feature_type_mapping[item[0]] = index
                index += 1
            return new_feature, feature_type_mapping

        def replace_value(data_type):
            index = 0
            feature_type_mapping = dict()
            node_type_set = np.unique(data_type)
            for item in node_type_set:
                data_type[data_type == item] = index
                feature_type_mapping[item] = index
                index += 1
            data_type = data_type.astype(np.int8)
            return data_type, feature_type_mapping

        def invert_dict(mapping):
            new_mapping = dict()
            for key, value in mapping.items():
                new_mapping[value] = key
            return new_mapping

        new_node_feat, self.node_feature_type_mapping = replace_dict_key(node_feat)
        new_edge_feat, self.edge_feature_type_mapping = replace_dict_key(edge_feat)
        new_graph_feat, self.graph_feature_type_mapping = replace_dict_key(graph_feat)
        new_node_type, self.node_type_mapping = replace_value(node_type)
        new_edge_type, self.edge_type_mapping = replace_value(edge_type)

        self.invert_node_type_mapping = invert_dict(self.node_type_mapping)
        self.invert_edge_type_mapping = invert_dict(self.edge_type_mapping)
        self.invert_node_feature_type_mapping = invert_dict(self.node_feature_type_mapping)
        self.invert_edge_feature_type_mapping = invert_dict(self.edge_feature_type_mapping)
        self.invert_graph_feature_type_mapping = invert_dict(self.graph_feature_type_mapping)

        return (new_node_feat, new_edge_feat, new_graph_feat, new_node_type, new_edge_type)


def save_graphs(path, graph_list, num_graphs_per_file=1, data_format="numpy"):
    """
    When init a graph, input parameter including: edges, node_feat, edge_feat, graph_feat, node_type, edge_type
    if do collate function, data of graph will be load into python layer
    but we consider to implement save graph in c++ layer, thus save to single graph_idx.npz firstly
    """

    def merge_into_dict(data, data_array, feature_type, prefix):
        for key, value in zip(feature_type, data_array):
            # shape each item should be [num_xxx, num_feature]
            data[prefix + str(key)] = value

    graph_data = dict()
    pre_idx = 0
    graph_num = len(graph_list)
    for idx, graph in enumerate(graph_list):
        graph_info = graph.graph_info()
        # currently input args of get_all_edges can only be int not list.
        edge_ids = graph.get_all_edges(graph_info["edge_type"][0])
        edges = np.array(graph.get_nodes_from_edges(edge_ids)).transpose()
        graph_data["graph_" + str(idx) + "_edges"] = edges

        # currently input args of get_all_nodes can only be int not list.
        node_ids = graph.get_all_nodes(graph_info["node_type"][0])
        if graph_info["node_feature_type"]:
            node_feat = graph.get_node_feature(node_ids, graph_info["node_feature_type"])
            merge_into_dict(graph_data, node_feat, graph_info["node_feature_type"], "graph_" + str(idx) + "_node_feat_")
        if graph_info["edge_feature_type"]:
            edge_feat = graph.get_edge_feature(edge_ids, graph_info["edge_feature_type"])
            merge_into_dict(graph_data, edge_feat, graph_info["edge_feature_type"], "graph_" + str(idx) + "_edge_feat_")
        if graph_info["graph_feature_type"]:
            graph_feat = graph.get_graph_feature(graph_info["graph_feature_type"])
            merge_into_dict(graph_data, graph_feat, graph_info["graph_feature_type"],
                            "graph_" + str(idx) + "_graph_feat_")

        # node_type and edge_type need to provide access interface, current unable to get
        if (idx + 1) % num_graphs_per_file == 0 or idx == (graph_num - 1):
            file_name = "graph_" + str(pre_idx) + "_" + str(idx) + ".npz"
            file_path = os.path.join(path, file_name)
            np.savez(file_path, **graph_data)
            graph_data = dict()
            pre_idx = idx + 1


def load_graphs(path, data_format="numpy", num_parallel_workers=1):
    """
    To be implemented in c++ layer, logic may similar as current implementation.
    """
    # consider add param like in graph param: working_mode, num_client ...
    files = [os.path.join(path, file_name) for file_name in os.listdir(path)]
    sorted(files)

    def get_feature_data(param_name, cols, graph_data):
        data_dict = dict()
        param_name = param_name + "_"
        for col in cols:
            if param_name in col:
                feature_type = col.split(param_name)[1]
                # reshape data with 2 dimension
                temp_data = graph_data[col]
                if len(temp_data.shape) == 1 and "graph_feat_" not in param_name:
                    temp_data = temp_data.reshape(temp_data.shape[0], 1)
                data_dict[feature_type] = temp_data
        return data_dict

    graphs = []
    for file in files:
        if not file.endswith("npz"):
            continue
        data = np.load(file)
        id_list = file.split("/")[-1].strip(".npz").split("_")
        ids = list(range(int(id_list[1]), int(id_list[2]) + 1))
        random.shuffle(ids)
        total_files = data.files
        for idx in ids:
            node_feat, edge_feat, graph_feat, node_type, edge_type = None, None, None, None, None
            keys = []
            prefix = "graph_" + str(idx) + "_"
            for item in total_files:
                if item.startswith(prefix):
                    keys.append(item)

            edges = data[prefix + "edges"]
            node_feat = get_feature_data(prefix + "node_feat", keys, data)
            edge_feat = get_feature_data(prefix + "edge_feat", keys, data)
            graph_feat = get_feature_data(prefix + "graph_feat", keys, data)

            if "node_type" in keys:
                node_type = data[prefix + "node_type"]
            if "edge_type" in keys:
                edge_type = data[prefix + "edge_type"]

            # consider graph been created in graph mode firstly
            graph = Graph(edges, node_feat, edge_feat, graph_feat, node_type, edge_type,
                          num_parallel_workers=num_parallel_workers)
            graphs.append(graph)
    return graphs


class _UsersDatasetTemplate:
    """
    Template for guiding user to create corresponding dataset(should inherit InMemoryGraphDataset when implemented).
    """

    class _ReInitTemplate:
        """
        Internal class _ReInitTemplate.
        """

        def __init__(self):
            pass

    def __new__(cls):
        # Before we overwrite '__init__' to user-defined '__init__',
        # we need make sure the '__init__' should be the basic version(_ReInitTemplate.__init__).
        basic_init_class = _UsersDatasetTemplate._ReInitTemplate()
        cls = object.__new__(cls)  # pylint: disable=W0642
        setattr(cls.__class__, "__init__", getattr(basic_init_class.__class__, "__init__"))
        return cls

    def __init__(self):
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        return 1

    def process(self):
        pass


class InMemoryGraphDataset(GeneratorDataset):
    """
    Basic Dataset for loading graph into memory.

    Recommended to Implement your own dataset with inheriting this class, and implement your own method like `process` ,
    `save` and `load` , refer source code of `ArgoverseDataset` for how to implement your own dataset. When init your
    own dataset like ArgoverseDataset, The executed process like follows. Check if there are already processed data
    under given `data_dir` , if so will call `load` method to load it directly, otherwise it will call `process` method
    to create graphs and call `save` method to save the graphs into `save_dir` .

    You can access graph in created dataset using `graphs = my_dataset.graphs` and also you can iterate dataset
    and get data using `my_dataset.create_tuple_iterator()` (in this way you need to implement methods like
    `__getitem__` and `__len__`), referring to the following example for detail. Note: we have overwritten the
    `__new__` method to reinitialize `__init__` internally, which means the user-defined `__new__` method won't work.

    Args:
        data_dir (str): directory for loading dataset, here contains origin format data and will be loaded in
            `process` method.
        save_dir (str): relative directory for saving processed dataset, this directory is under `data_dir` .
            Default: './processed'.
        column_names (Union[str, list[str]], optional): single column name or list of column names of the dataset,
            num of column name should be equal to num of item in return data when implement method like `__getitem__` .
            Default: 'graph'.
        num_samples (int, optional): The number of samples to be included in the dataset. Default: None, all samples.
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. This parameter can only be
            specified when the implemented dataset has a random access attribute ( `__getitem__` ). Default: None.
        num_shards (int, optional): Number of shards that the dataset will be divided into. Default: None.
            When this argument is specified, `num_samples` reflects the max
            sample number of per shard.
        shard_id (int, optional): The shard ID within `num_shards` . Default: None. This argument must be specified only
            when `num_shards` is also specified.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: True.
        max_rowsize(int, optional): Maximum size of row in MB that is used for shared memory allocation to copy
            data between processes. This is only used if python_multiprocessing is set to True. Default: 6 MB.

    Raises:
        TypeError: If `data_dir` is not of type str.
        TypeError: If `save_dir` is not of type str.
        TypeError: If `num_parallel_workers` is not of type int.
        TypeError: If `shuffle` is not of type bool.
        TypeError: If `python_multiprocessing` is not of type bool.
        TypeError: If `perf_mode` is not of type bool.
        RuntimeError: If `data_dir` is not valid or does not exit.
        RuntimeError: If `num_shards` is specified but `shard_id` is None.
        RuntimeError: If `shard_id` is specified but `num_shards` is None.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> from mindspore.dataset import InMemoryGraphDataset, Graph
        >>>
        >>> class MyDataset(InMemoryGraphDataset):
        ...     def __init__(self, data_dir):
        ...         super().__init__(data_dir)
        ...
        ...     def process(self):
        ...         # create graph with loading data in given data_dir
        ...         # here create graph with numpy array directly instead
        ...         edges = np.array([[0, 1], [1, 2]])
        ...         graph = Graph(edges=edges)
        ...         self.graphs.append(graph)
        ...
        ...     def __getitem__(self, index):
        ...         # this method and '__len__' method are required when iterating created dataset
        ...         graph = self.graphs[index]
        ...         return graph.get_all_edges('0')
        ...
        ...     def __len__(self):
        ...         return len(self.graphs)
    """

    def __init__(self, data_dir, save_dir="./processed", column_names="graph", num_samples=None, num_parallel_workers=1,
                 shuffle=None, num_shards=None, shard_id=None, python_multiprocessing=True, max_rowsize=6):
        self.graphs = []
        self.data_dir = data_dir
        self.save_dir = save_dir
        self.processed_path = os.path.join(self.data_dir, self.save_dir)
        self.processed = False
        if 'process' in self.__class__.__dict__:
            self._process()
        if self.processed:
            self.load()

        source = _UsersDatasetTemplate()
        for k, v in self.__dict__.items():
            setattr(source, k, v)
        for k, v in self.__class__.__dict__.items():
            if k == "__new__":
                # The user-defined '__new__' is skipped.
                continue
            setattr(source.__class__, k, getattr(self.__class__, k))
        super().__init__(source, column_names=column_names, num_samples=num_samples,
                         num_parallel_workers=num_parallel_workers, shuffle=shuffle, num_shards=num_shards,
                         shard_id=shard_id, python_multiprocessing=python_multiprocessing, max_rowsize=max_rowsize)

    def process(self):
        """
        Process method based on origin dataset, override this method in your our dataset class.
        """
        raise NotImplementedError("'process' method should be implemented in your own logic.")

    def save(self):
        """
        Save processed data into disk in numpy.npz format, you can also override this method in your dataset class.
        """
        save_graphs(self.processed_path, self.graphs)

    def load(self):
        """
        Load data from given(processed) path, you can also override this method in your dataset class.
        """
        self.graphs = load_graphs(self.processed_path, num_parallel_workers=1)

    def _process(self):
        # file has been processed and saved into processed_path
        if not os.path.isdir(self.processed_path):
            os.makedirs(self.processed_path, exist_ok=True)
        elif os.listdir(self.processed_path):
            self.processed = True
            return
        self.process()
        self.save()


class ArgoverseDataset(InMemoryGraphDataset):
    """
    Load argoverse dataset and create graph.

    Here argoverse dataset is public dataset for autonomous driving, current implement `ArgoverseDataset` is mainly for
    loading Motion Forecasting Dataset in argoverse dataset, recommend to visit official website for more detail:
    https://www.argoverse.org/av1.html#download-link.

    Args:
        data_dir (str): directory for loading dataset, here contains origin format data and will be loaded in
            `process` method.
        column_names (Union[str, list[str]], optional): single column name or list of column names of the dataset.
            Default: "graph". Num of column name should be equal to num of item in return data when implement method
            like `__getitem__`, recommend to specify it with
            `column_names=["edge_index", "x", "y", "cluster", "valid_len", "time_step_len"]` like the following example.
        num_parallel_workers (int, optional): Number of subprocesses used to fetch the dataset in parallel. Default: 1.
        shuffle (bool, optional): Whether or not to perform shuffle on the dataset. This parameter can only be
            specified when the implemented dataset has a random access attribute ( `__getitem__` ). Default: None.
        python_multiprocessing (bool, optional): Parallelize Python operations with multiple worker process. This
            option could be beneficial if the Python operation is computational heavy. Default: True.
        perf_mode(bool, optional): mode for obtaining higher performance when iterate created dataset(will call
            `__getitem__` method in this process). Default True, will save all the data in graph
            (like edge index, node feature and graph feature) into graph feature.

    Raises:
        TypeError: If `data_dir` is not of type str.
        TypeError: If `num_parallel_workers` is not of type int.
        TypeError: If `shuffle` is not of type bool.
        TypeError: If `python_multiprocessing` is not of type bool.
        TypeError: If `perf_mode` is not of type bool.
        RuntimeError: If `data_dir` is not valid or does not exit.
        ValueError: If `num_parallel_workers` exceeds the max thread numbers.

    Examples:
        >>> from mindspore.dataset import ArgoverseDataset
        >>>
        >>> argoverse_dataset_dir = "/path/to/argoverse_dataset_directory"
        >>> graph_dataset = ArgoverseDataset(data_dir=argoverse_dataset_dir,
        ...                                  column_names=["edge_index", "x", "y", "cluster", "valid_len",
        ...                                                "time_step_len"])
        >>> for item in graph_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        ...     pass

    About Argoverse Dataset:

    Argverse is the first dataset containing high-precision maps, which contains 290KM high-precision map data with
    geometric shape and semantic information.

    You can unzip the dataset files into the following structure and read by MindSpore's API:

    .. code-block::

        .
        └── argoverse_dataset_dir
            ├── train
            │    ├──...
            ├── val
            │    └──...
            ├── test
            │    └──...

    Citation:

    .. code-block::

        @inproceedings{Argoverse,
        author     = {Ming-Fang Chang and John W Lambert and Patsorn Sangkloy and Jagjeet Singh
                   and Slawomir Bak and Andrew Hartnett and De Wang and Peter Carr
                   and Simon Lucey and Deva Ramanan and James Hays},
        title      = {Argoverse: 3D Tracking and Forecasting with Rich Maps},
        booktitle  = {Conference on Computer Vision and Pattern Recognition (CVPR)},
        year       = {2019}
        }
    """

    def __init__(self, data_dir, column_names="graph", num_parallel_workers=1, shuffle=None,
                 python_multiprocessing=True, perf_mode=True):
        # For high performance, here we store edge_index into graph_feature directly
        if not isinstance(perf_mode, bool):
            raise TypeError("Type of 'perf_mode' should be bool, but got {}.".format(type(perf_mode)))
        self.perf_mode = perf_mode
        super().__init__(data_dir=data_dir, column_names=column_names, shuffle=shuffle,
                         num_parallel_workers=num_parallel_workers, python_multiprocessing=python_multiprocessing)

    def __getitem__(self, index):
        graph = self.graphs[index]
        if self.perf_mode:
            return tuple(graph.get_graph_feature(
                feature_types=["edge_index", "x", "y", "cluster", "valid_len", "time_step_len"]))

        graph_info = graph.graph_info()
        all_nodes = graph.get_all_nodes(graph_info["node_type"][0])
        edge_ids = graph.get_all_edges(graph_info["edge_type"][0])
        edge_index = np.array(graph.get_nodes_from_edges(edge_ids)).transpose()
        x = graph.get_node_feature(all_nodes, feature_types=["x"])[0]
        graph_feature = graph.get_graph_feature(feature_types=["y", "cluster", "valid_len", "time_step_len"])
        y, cluster, valid_len, time_step_len = graph_feature

        return edge_index, x, y, cluster, valid_len, time_step_len

    def __len__(self):
        return len(self.graphs)

    def process(self):
        """
        Process method for argoverse dataset, here we load original dataset and create a lot of graphs based on it.
        Pre-processed method mainly refers to: https://github.com/xk-huang/yet-another-vectornet/blob/master/dataset.py.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError("Import pandas failed, recommend to install pandas with pip.")

        def get_edge_full_connection(node_num, start_index=0):
            """
            Obtain edge_index with shape (2, edge_num)
            """
            edges = np.empty((2, 0))
            end = np.arange(node_num, dtype=np.int64)
            for idx in range(node_num):
                begin = np.ones(node_num, dtype=np.int64) * idx
                edges = np.hstack((edges, np.vstack(
                    (np.hstack([begin[:idx], begin[idx + 1:]]), np.hstack([end[:idx], end[idx + 1:]])))))
            edges = edges + start_index

            return edges.astype(np.int64), node_num + start_index

        file_path = [os.path.join(self.data_dir, file_name) for file_name in os.listdir(self.data_dir)]
        sorted(file_path)

        valid_len_list = []
        data_list = []
        for data_p in file_path:
            if not data_p.endswith('pkl'):
                continue
            x_list, edge_index_list = [], []
            data = pd.read_pickle(data_p)
            input_features = data['POLYLINE_FEATURES'].values[0]
            basic_len = data['TARJ_LEN'].values[0]
            cluster = input_features[:, -1].reshape(-1).astype(np.int32)
            valid_len_list.append(cluster.max())
            y = data['GT'].values[0].reshape(-1).astype(np.float32)

            traj_id_mask = data["TRAJ_ID_TO_MASK"].values[0]
            lane_id_mask = data['LANE_ID_TO_MASK'].values[0]
            start_idx = 0

            for _, mask in traj_id_mask.items():
                feature = input_features[mask[0]:mask[1]]
                temp_edge, start_idx = get_edge_full_connection(
                    feature.shape[0], start_idx)
                x_list.append(feature)
                edge_index_list.append(temp_edge)

            for _, mask in lane_id_mask.items():
                feature = input_features[mask[0] + basic_len: mask[1] + basic_len]
                temp_edge, start_idx = get_edge_full_connection(
                    feature.shape[0], start_idx)
                x_list.append(feature)
                edge_index_list.append(temp_edge)
            edge_index = np.hstack(edge_index_list)
            x = np.vstack(x_list)
            data_list.append([x, y, cluster, edge_index])

        graphs = []
        pad_to_index = np.max(valid_len_list)
        feature_len = data_list[0][0].shape[1]
        for index, item in enumerate(data_list):
            item[0] = np.vstack(
                [item[0], np.zeros((pad_to_index - item[-2].max(), feature_len), dtype=item[0].dtype)])
            item[-2] = np.hstack(
                [item[2], np.arange(item[-2].max() + 1, pad_to_index + 1)])

            if self.perf_mode:
                graph_feature = {"edge_index": item[3], "x": item[0], "y": item[1], "cluster": item[2],
                                 "valid_len": np.array([valid_len_list[index]]),
                                 "time_step_len": np.array([pad_to_index + 1])}
                g_data = Graph(edges=item[3], graph_feat=graph_feature)
            else:
                node_feature = {"x": item[0]}
                graph_feature = {"y": item[1], "cluster": item[2], "valid_len": np.array([valid_len_list[index]]),
                                 "time_step_len": np.array([pad_to_index + 1])}
                g_data = Graph(edges=item[3], node_feat=node_feature, graph_feat=graph_feature)
            graphs.append(g_data)
        self.graphs = graphs
