import os
import random
import time
from multiprocessing import Process
import numpy as np
from mindspore import log as logger
from mindspore.dataset import Graph
from mindspore.dataset import ArgoverseDataset


def test_create_graph_with_edges():
    """
    Feature: Graph
    Description: test create Graph with loading edge, node_feature, edge_feature
    Expectation: Output value equals to the expected
    """
    edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
    node_feat = {"label": np.array([[0], [1], [2]], dtype=np.int32)}
    edge_feat = {"feature": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    g = Graph(edges, node_feat, edge_feat)

    graph_info = g.graph_info()
    assert graph_info['node_type'] == ['0']
    assert graph_info['edge_type'] == ['0']
    assert graph_info['node_num'] == {'0': 3}
    assert graph_info['edge_num'] == {'0': 2}
    assert graph_info['node_feature_type'] == ['label']
    assert graph_info['edge_feature_type'] == ['feature']

    all_nodes = g.get_all_nodes('0')
    assert all_nodes.tolist() == [0, 1, 2]
    all_edges = g.get_all_edges('0')
    assert all_edges.tolist() == [0, 1]
    node_feature = g.get_node_feature([0, 1], ["label"])
    assert node_feature[0].tolist() == [0, 1]
    edge_feature = g.get_edge_feature([0], ["feature"])
    assert edge_feature[0].tolist() == [1, 2]


def start_graph_server_with_array(server_port):
    """
    start graph server.
    """
    edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
    node_feat = {"label": np.array([[0], [1], [2]], dtype=np.int32)}
    edge_feat = {"feature": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    graph_feat = {"feature_1": np.array([1, 2, 3, 4, 5], dtype=np.int32),
                  "feature_2": np.array([11, 12, 13, 14, 15], dtype=np.int32)}
    Graph(edges, node_feat, edge_feat, graph_feat, working_mode='server', port=server_port)


def test_server_mode_with_array():
    """
    Feature: Graph
    Description: Test Graph distributed
    Expectation: Output equals to the expected output
    """
    asan = os.environ.get('ASAN_OPTIONS')
    if asan:
        logger.info("skip the Graph distributed when asan mode")
        return

    server_port = random.randint(10000, 60000)
    p1 = Process(target=start_graph_server_with_array, args=(server_port,))
    p1.start()
    time.sleep(5)

    edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
    node_feat = {"label": np.array([[0], [1], [2]], dtype=np.int32)}
    edge_feat = {"feature": np.array([[1, 2], [3, 4]], dtype=np.int32)}
    graph_feat = {"feature_1": np.array([1, 2, 3, 4, 5], dtype=np.int32),
                  "feature_2": np.array([11, 12, 13, 14, 15], dtype=np.int32)}
    g = Graph(edges, node_feat, edge_feat, graph_feat, working_mode='client', port=server_port)

    all_nodes = g.get_all_nodes('0')
    assert all_nodes.tolist() == [0, 1, 2]
    all_edges = g.get_all_edges('0')
    assert all_edges.tolist() == [0, 1]
    node_feature = g.get_node_feature([0, 1], ["label"])
    assert node_feature[0].tolist() == [0, 1]
    edge_feature = g.get_edge_feature([0], ["feature"])
    assert edge_feature[0].tolist() == [1, 2]
    graph_feature = g.get_graph_feature(["feature_1"])
    assert graph_feature[0].tolist() == [1, 2, 3, 4, 5]


def test_graph_feature_local():
    """
    Feature: Graph
    Description: Test load Graph feature in local mode
    Expectation: Output equals to the expected output
    """
    edges = np.array([[1, 2], [0, 1]], dtype=np.int32)
    graph_feat = {"feature_1": np.array([1, 2, 3, 4, 5], dtype=np.int32),
                  "feature_2": np.array([11, 12, 13, 14, 15], dtype=np.int32)}

    g = Graph(edges, graph_feat=graph_feat)
    graph_feature = g.get_graph_feature(["feature_1"])
    assert graph_feature[0].tolist() == [1, 2, 3, 4, 5]


def test_argoverse_dataset():
    """
    Feature: Graph
    Description: Test self-implemented dataset which inherit InMemoryGraphDataset
    Expectation: Output equals to the expected output
    """
    data_dir = "../data/dataset/testArgoverse"
    graph_dataset = ArgoverseDataset(data_dir,
                                     column_names=["edge_index", "x", "y", "cluster", "valid_len", "time_step_len"])
    for item in graph_dataset.create_dict_iterator(output_numpy=True, num_epochs=1):
        keys = list(item.keys())
        assert keys == ["edge_index", "x", "y", "cluster", "valid_len", "time_step_len"]


if __name__ == "__main__":
    test_create_graph_with_edges()
    test_graph_feature_local()
    test_server_mode_with_array()
    test_argoverse_dataset()
