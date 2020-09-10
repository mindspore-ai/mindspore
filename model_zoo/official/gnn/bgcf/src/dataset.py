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
preprocess raw data; generate batched data and sample neighbors on graph for training and test;
Amazon Beauty datasets are supported by our example, the original versions of these datasets are as follows:
    @article{Amazon Beauty,
    title     = {Ups and Downs: Modeling the Visual Evolution of Fashion Trends with One-Class Collaborative Filtering},
    author    = {R. He, J. McAuley},
    journal   = {WWW},
    year      = {2016},
    url       = {http://jmcauley.ucsd.edu/data/amazon}
    }
"""
import numpy as np
import mindspore.dataset as ds


class RandomBatchedSampler(ds.Sampler):
    """RandomBatchedSampler generate random sequence without replacement in a batched manner"""

    sampled_graph_index = 0

    def __init__(self, index_range, num_edges_per_sample):
        super().__init__()
        self.index_range = index_range
        self.num_edges_per_sample = num_edges_per_sample

    def __iter__(self):
        self.sampled_graph_index += 1
        indices = [i for i in range(self.index_range)]
        np.random.shuffle(indices)
        for i in range(0, self.index_range, self.num_edges_per_sample):
            if i + self.num_edges_per_sample <= self.index_range:
                result = indices[i: i + self.num_edges_per_sample]
                result.append(self.sampled_graph_index)
                yield result


class TrainGraphDataset():
    """Sample node neighbors on graphs for training"""

    def __init__(self, train_graph, sampled_graphs, batch_num, num_samples, num_bgcn_neigh, num_neg):
        self.g = train_graph
        self.batch_num = batch_num
        self.sampled_graphs = sampled_graphs
        self.sampled_graph_num = len(sampled_graphs)
        self.num_samples = num_samples
        self.num_bgcn_neigh = num_bgcn_neigh
        self.num_neg = num_neg

    def __len__(self):
        return self.g.graph_info()['edge_num'][0] // self.batch_num

    def __getitem__(self, index):
        """
        Sample negative items with their neighbors, user neighbors, pos item neighbors
        based on the user-item pairs
        """
        sampled_graph_index = index[-1] % self.sampled_graph_num
        index = index[0:-1]
        train_graph = self.g
        sampled_graph = self.sampled_graphs[sampled_graph_index]

        rating = train_graph.get_nodes_from_edges(index.astype(np.int32))
        users = rating[:, 0]

        u_group_nodes = train_graph.get_sampled_neighbors(
            node_list=users, neighbor_nums=[1], neighbor_types=[0])
        pos_users = u_group_nodes[:, 1]
        u_group_nodes = np.concatenate((users, pos_users), axis=0)
        u_group_nodes = u_group_nodes.reshape(-1,).tolist()
        u_neighs = train_graph.get_sampled_neighbors(
            node_list=u_group_nodes, neighbor_nums=[self.num_samples], neighbor_types=[1])
        u_neighs = u_neighs[:, 1:]
        u_gnew_neighs = sampled_graph.get_sampled_neighbors(
            node_list=u_group_nodes, neighbor_nums=[self.num_bgcn_neigh], neighbor_types=[1])
        u_gnew_neighs = u_gnew_neighs[:, 1:]

        items = rating[:, 1]
        i_group_nodes = train_graph.get_sampled_neighbors(
            node_list=items, neighbor_nums=[1], neighbor_types=[1])
        pos_items = i_group_nodes[:, 1]
        i_group_nodes = np.concatenate((items, pos_items), axis=0)
        i_group_nodes = i_group_nodes.reshape(-1,).tolist()
        i_neighs = train_graph.get_sampled_neighbors(
            node_list=i_group_nodes, neighbor_nums=[self.num_samples], neighbor_types=[0])
        i_neighs = i_neighs[:, 1:]
        i_gnew_neighs = sampled_graph.get_sampled_neighbors(
            node_list=i_group_nodes, neighbor_nums=[self.num_bgcn_neigh], neighbor_types=[0])
        i_gnew_neighs = i_gnew_neighs[:, 1:]

        neg_item_id = train_graph.get_neg_sampled_neighbors(
            node_list=users, neg_neighbor_num=self.num_neg, neg_neighbor_type=1)
        neg_item_id = neg_item_id[:, 1:]
        neg_group_nodes = neg_item_id.reshape(-1,)
        neg_neighs = train_graph.get_sampled_neighbors(
            node_list=neg_group_nodes, neighbor_nums=[self.num_samples], neighbor_types=[0])
        neg_neighs = neg_neighs[:, 1:]
        neg_gnew_neighs = sampled_graph.get_sampled_neighbors(
            node_list=neg_group_nodes, neighbor_nums=[self.num_bgcn_neigh], neighbor_types=[0])
        neg_gnew_neighs = neg_gnew_neighs[:, 1:]

        return users, items, neg_item_id, pos_users, pos_items, u_group_nodes, u_neighs, u_gnew_neighs, \
               i_group_nodes, i_neighs, i_gnew_neighs, neg_group_nodes, neg_neighs, neg_gnew_neighs


class TestGraphDataset():
    """Sample node neighbors on graphs for test"""

    def __init__(self, g, sampled_graphs, num_samples, num_bgcn_neigh, num_neg):
        self.g = g
        self.sampled_graphs = sampled_graphs
        self.sampled_graph_index = 0
        self.num_samples = num_samples
        self.num_bgcn_neigh = num_bgcn_neigh
        self.num_neg = num_neg
        self.num_user = self.g.graph_info()["node_num"][0]
        self.num_item = self.g.graph_info()["node_num"][1]

    def random_select_sampled_graph(self):
        self.sampled_graph_index = np.random.randint(len(self.sampled_graphs))

    def get_user_sapmled_neighbor(self):
        """Sample all users neighbors for test"""
        users = np.arange(self.num_user, dtype=np.int32)
        u_neighs = self.g.get_sampled_neighbors(
            node_list=users, neighbor_nums=[self.num_samples], neighbor_types=[1])
        u_neighs = u_neighs[:, 1:]
        sampled_graph = self.sampled_graphs[self.sampled_graph_index]
        u_gnew_neighs = sampled_graph.get_sampled_neighbors(
            node_list=users, neighbor_nums=[self.num_bgcn_neigh], neighbor_types=[1])
        u_gnew_neighs = u_gnew_neighs[:, 1:]
        return u_neighs, u_gnew_neighs

    def get_item_sampled_neighbor(self):
        """Sample all items neighbors for test"""
        items = np.arange(self.num_user, self.num_user + self.num_item, dtype=np.int32)
        i_neighs = self.g.get_sampled_neighbors(
            node_list=items, neighbor_nums=[self.num_samples], neighbor_types=[0])
        i_neighs = i_neighs[:, 1:]

        sampled_graph = self.sampled_graphs[self.sampled_graph_index]
        i_gnew_neighs = sampled_graph.get_sampled_neighbors(
            node_list=items, neighbor_nums=[self.num_bgcn_neigh], neighbor_types=[0])
        i_gnew_neighs = i_gnew_neighs[:, 1:]
        return i_neighs, i_gnew_neighs


def load_graph(data_path):
    """Load train graph, test graph and sampled graph"""
    train_graph = ds.GraphData(
        data_path + "/train_mr", num_parallel_workers=8)

    test_graph = ds.GraphData(
        data_path + "/test_mr", num_parallel_workers=8)

    sampled_graph_list = []
    for i in range(0, 5):
        sampled_graph = ds.GraphData(
            data_path + "/sampled" + str(i) + "_mr", num_parallel_workers=8)
        sampled_graph_list.append(sampled_graph)

    return train_graph, test_graph, sampled_graph_list


def create_dataset(train_graph, sampled_graph_list, num_workers, batch_size=32, repeat_size=1,
                   num_samples=40, num_bgcn_neigh=20, num_neg=10):
    """Data generator for training"""
    edge_num = train_graph.graph_info()['edge_num'][0]
    out_column_names = ["users", "items", "neg_item_id", "pos_users", "pos_items", "u_group_nodes", "u_neighs",
                        "u_gnew_neighs", "i_group_nodes", "i_neighs", "i_gnew_neighs", "neg_group_nodes",
                        "neg_neighs", "neg_gnew_neighs"]
    train_graph_dataset = TrainGraphDataset(
        train_graph, sampled_graph_list, batch_size, num_samples, num_bgcn_neigh, num_neg)
    dataset = ds.GeneratorDataset(source=train_graph_dataset, column_names=out_column_names,
                                  sampler=RandomBatchedSampler(edge_num, batch_size), num_parallel_workers=num_workers)
    dataset = dataset.repeat(repeat_size)

    return dataset
