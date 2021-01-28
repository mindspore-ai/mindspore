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

import os
import random
import time
from multiprocessing import Process
import numpy as np
import mindspore.dataset as ds
from mindspore import log as logger
from mindspore.dataset.engine import SamplingStrategy

DATASET_FILE = "../data/mindrecord/testGraphData/testdata"


def graphdata_startserver(server_port):
    """
    start graphdata server
    """
    logger.info('test start server.\n')
    ds.GraphData(DATASET_FILE, 1, 'server', port=server_port)


class RandomBatchedSampler(ds.Sampler):
    # RandomBatchedSampler generate random sequence without replacement in a batched manner
    def __init__(self, index_range, num_edges_per_sample):
        super().__init__()
        self.index_range = index_range
        self.num_edges_per_sample = num_edges_per_sample

    def __iter__(self):
        indices = [i+1 for i in range(self.index_range)]
        # Reset random seed here if necessary
        # random.seed(0)
        random.shuffle(indices)
        for i in range(0, self.index_range, self.num_edges_per_sample):
            # Drop reminder
            if i + self.num_edges_per_sample <= self.index_range:
                yield indices[i: i + self.num_edges_per_sample]


class GNNGraphDataset():
    def __init__(self, g, batch_num):
        self.g = g
        self.batch_num = batch_num

    def __len__(self):
        # Total sample size of GNN dataset
        # In this case, the size should be total_num_edges/num_edges_per_sample
        return self.g.graph_info()['edge_num'][0] // self.batch_num

    def __getitem__(self, index):
        # index will be a list of indices yielded from RandomBatchedSampler
        # Fetch edges/nodes/samples/features based on indices
        nodes = self.g.get_nodes_from_edges(index.astype(np.int32))
        nodes = nodes[:, 0]
        neg_nodes = self.g.get_neg_sampled_neighbors(
            node_list=nodes, neg_neighbor_num=3, neg_neighbor_type=1)
        nodes_neighbors = self.g.get_sampled_neighbors(node_list=nodes, neighbor_nums=[
            2, 2], neighbor_types=[2, 1], strategy=SamplingStrategy.RANDOM)
        neg_nodes_neighbors = self.g.get_sampled_neighbors(node_list=neg_nodes[:, 1:].reshape(-1), neighbor_nums=[2, 2],
                                                           neighbor_types=[2, 1], strategy=SamplingStrategy.EDGE_WEIGHT)
        nodes_neighbors_features = self.g.get_node_feature(
            node_list=nodes_neighbors, feature_types=[2, 3])
        neg_neighbors_features = self.g.get_node_feature(
            node_list=neg_nodes_neighbors, feature_types=[2, 3])
        return nodes_neighbors, neg_nodes_neighbors, nodes_neighbors_features[0], neg_neighbors_features[1]


def test_graphdata_distributed():
    """
    Test distributed
    """
    ASAN = os.environ.get('ASAN_OPTIONS')
    if ASAN:
        logger.info("skip the graphdata distributed when asan mode")
        return

    logger.info('test distributed.\n')

    server_port = random.randint(10000, 60000)

    p1 = Process(target=graphdata_startserver, args=(server_port,))
    p1.start()
    time.sleep(5)

    g = ds.GraphData(DATASET_FILE, 1, 'client', port=server_port)
    nodes = g.get_all_nodes(1)
    assert nodes.tolist() == [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
    row_tensor = g.get_node_feature(nodes.tolist(), [1, 2, 3])
    assert row_tensor[0].tolist() == [[0, 1, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 1, 1, 0], [0, 0, 0, 0, 0],
                                      [1, 1, 0, 1, 0], [0, 0, 0, 0, 1], [0, 1, 0, 0, 0], [0, 0, 0, 1, 1],
                                      [0, 1, 1, 0, 0], [0, 1, 0, 1, 0]]
    assert row_tensor[2].tolist() == [1, 2, 3, 1, 4, 3, 5, 3, 5, 4]

    edges = g.get_all_edges(0)
    assert edges.tolist() == [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
                              21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]
    features = g.get_edge_feature(edges, [1, 2])
    assert features[0].tolist() == [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,
                                    0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0]

    batch_num = 2
    edge_num = g.graph_info()['edge_num'][0]
    out_column_names = ["neighbors", "neg_neighbors", "neighbors_features", "neg_neighbors_features"]
    dataset = ds.GeneratorDataset(source=GNNGraphDataset(g, batch_num), column_names=out_column_names,
                                  sampler=RandomBatchedSampler(edge_num, batch_num), num_parallel_workers=4,
                                  python_multiprocessing=False)
    dataset = dataset.repeat(2)
    itr = dataset.create_dict_iterator(num_epochs=1, output_numpy=True)
    i = 0
    for data in itr:
        assert data['neighbors'].shape == (2, 7)
        assert data['neg_neighbors'].shape == (6, 7)
        assert data['neighbors_features'].shape == (2, 7)
        assert data['neg_neighbors_features'].shape == (6, 7)
        i += 1
    assert i == 40


if __name__ == '__main__':
    test_graphdata_distributed()
