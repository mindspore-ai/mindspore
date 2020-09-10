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
Preprocess data.
"""
import time
import random
import argparse
import pickle as pkl
from copy import deepcopy
from functools import partial
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from sklearn.model_selection import train_test_split


def load_pickle(path, name):
    """Load pickle"""
    with open(path + name, 'rb') as f:
        return pkl.load(f, encoding='latin1')


def generate_inverse_mapping(data_list):
    """Generate inverse id map"""
    ds_matrix_mapping = dict()
    for inner_id, true_id in enumerate(data_list):
        ds_matrix_mapping[true_id] = inner_id
    return ds_matrix_mapping


def convert_to_inner_index(user_records, user_mapping, item_mapping):
    """Convert real id to inner id"""
    inner_user_records = []
    user_inverse_mapping = generate_inverse_mapping(user_mapping)
    item_inverse_mapping = generate_inverse_mapping(item_mapping)
    for user_id, _ in enumerate(user_mapping):
        real_user_id = user_mapping[user_id]
        item_list = list(user_records[real_user_id])
        for index, real_item_id in enumerate(item_list):
            item_list[index] = item_inverse_mapping[real_item_id]
        inner_user_records.append(item_list)
    return inner_user_records, user_inverse_mapping, item_inverse_mapping


def split_data_randomly(user_records, test_ratio, seed=0):
    """Split data"""
    print('seed %d ' % seed)
    train_set = []
    test_set = []
    for _, item_list in enumerate(user_records):
        tmp_train_sample, tmp_test_sample = train_test_split(
            item_list, test_size=test_ratio, random_state=seed)

        train_sample = []
        for place in item_list:
            if place not in tmp_test_sample:
                train_sample.append(place)

        test_sample = []
        for place in tmp_test_sample:
            if place not in tmp_train_sample:
                test_sample.append(place)

        train_set.append(train_sample)
        test_set.append(test_sample)

    return train_set, test_set


def create_adj_matrix(train_matrix):
    """Create adj matrix"""
    user2item, item2user = {}, {}
    user_item_ratings = train_matrix.toarray()
    for i, _ in enumerate(user_item_ratings):
        neigh_items = np.where(user_item_ratings[i] != 0)[0].tolist()
        user2item[i] = set(neigh_items)
    item_user_ratings = user_item_ratings.transpose()
    for j, _ in enumerate(item_user_ratings):
        neigh_users = np.where(item_user_ratings[j] != 0)[0].tolist()
        item2user[j] = set(neigh_users)
    return user2item, item2user


def generate_rating_matrix(train_set, num_users, num_items, user_shift=0, item_shift=0):
    """Generate rating matrix"""
    row = []
    col = []
    data = []
    for user_id, article_list in enumerate(train_set):
        for article in article_list:
            row.append(user_id + user_shift)
            col.append(article + item_shift)
            data.append(1)
    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix(
        (data, (row, col)), shape=(num_users, num_items))
    return rating_matrix


def flatten(distance, adj, thre=10):
    """Flatten the distance matrix for the smoother sampling"""
    print('start flattening the dataset with threshold = {}'.format(thre))
    top_ids = np.argsort(distance, 1)[:, -thre:]

    flat_distance = np.zeros_like(distance)
    values = 1 / thre
    for i, _ in enumerate(flat_distance):
        adj_len = len(adj[i])
        if adj_len == 0 or adj_len > thre:
            flat_distance[i][top_ids[i]] = values
        else:
            flat_distance[i][top_ids[i][thre - adj_len:]] = 1 / adj_len
    return flat_distance


def sample_graph_copying(node_neighbors_dict, distances, epsilon=0.01, seed=0, set_seed=False):
    """node copying node by node"""
    if set_seed:
        np.random.seed(seed)
        random.seed(seed)

    N = len(distances)

    sampled_graph = dict()
    nodes = np.arange(0, N).astype(np.int)

    for i in range(N):
        if random.uniform(0, 1) < 1 - epsilon:
            sampled_node = np.random.choice(nodes, 1, p=distances[i])
        else:
            sampled_node = [i]

        sampled_graph[i] = node_neighbors_dict[sampled_node[0]]

    return sampled_graph


def remove_infrequent_users(data, min_counts=10):
    """Remove infrequent users"""
    df = deepcopy(data)
    counts = df['user_id'].value_counts()
    df = df[df["user_id"].isin(counts[counts >= min_counts].index)]

    print("users with < {} interactoins are removed".format(min_counts))
    return df


def remove_infrequent_items(data, min_counts=5):
    """Remove infrequent items"""
    df = deepcopy(data)
    counts = df['item_id'].value_counts()
    df = df[df["item_id"].isin(counts[counts >= min_counts].index)]

    print("items with < {} interactoins are removed".format(min_counts))
    return df


def save_obj(obj, data_path, name):
    """Save object"""
    with open(data_path + "/" + name + '.pkl', 'wb') as f:
        pkl.dump(obj, f)


def preprocess_data(data_path, data_name):
    """Preprocess data"""
    rating_file = 'ratings_{}.csv'.format(data_name)
    col_names = ['user_id', 'item_id', 'rating', 'timestamp']
    data_records = pd.read_csv(data_path + "/" + rating_file, sep=',', names=col_names, engine='python')

    data_records.loc[data_records.rating != 0, 'rating'] = 1
    data_records = data_records[data_records.rating > 0]
    filtered_data = remove_infrequent_users(data_records, 10)
    filtered_data = remove_infrequent_items(filtered_data, 10)

    data = filtered_data.groupby('user_id')['item_id'].apply(list)
    unique_data = filtered_data.groupby('user_id')['item_id'].nunique()
    data = data[unique_data[unique_data >= 5].index]

    user_item_dict = data.to_dict()
    user_mapping = []
    item_set = set()
    for user_id, item_list in data.iteritems():
        user_mapping.append(user_id)
        for item_id in item_list:
            item_set.add(item_id)
    item_mapping = list(item_set)
    return user_item_dict, user_mapping, item_mapping


def iou_set(set1, set2):
    """Calculate iou_set """
    union = set1.union(set2)
    return len(set1.intersection(set2)) / len(union) if union else 0


def build_func(train_set, data):
    """Build function"""
    res = []
    res.append([iou_set(set(train_set), x) for x in data.values()])
    return res


def build_distance_mp_map(train_set, u_adj_list, v_adj_list, num_workers=5, tag='user', norm=True):
    """Build distance matrix"""
    start = time.time()
    pool = Pool(processes=num_workers)

    if tag == 'user':
        results = pool.map_async(partial(build_func, data=u_adj_list), train_set)

    if tag == 'item':
        results = pool.map_async(partial(build_func, data=v_adj_list), train_set)

    results.wait()

    pool.close()
    pool.join()

    distances = np.array(results.get()).squeeze(1)
    np.fill_diagonal(distances, 0)
    print('=== info: elapsed time with mp for building ' + tag + ' distance matrix: ', time.time() - start)

    for i, _ in enumerate(distances):
        if sum(distances[i]) == 0:
            distances[i] = 1.

    if norm:
        distances = distances / np.sum(distances, axis=1).reshape(-1, 1)
    distances.astype(np.float16)
    return distances


def trans(src_path, data_name, out_path):
    """Convert into MindSpore data"""
    print('=== loading datasets')
    user_records, user_mapping, item_mapping = preprocess_data(src_path, data_name)
    inner_data_records, user_inverse_mapping, \
    item_inverse_mapping = convert_to_inner_index(
        user_records, user_mapping, item_mapping)

    test_ratio = 0.2
    train_set, test_set = split_data_randomly(
        inner_data_records, test_ratio, seed=0)
    train_matrix = generate_rating_matrix(
        train_set, len(user_mapping), len(item_mapping))
    u_adj_list, v_adj_list = create_adj_matrix(train_matrix)
    num_user, num_item = train_matrix.shape

    print('=== building user-user grpah and item-item graph')
    num_self_neigh = 10
    user_user_graph = kneighbors_graph(train_matrix, num_self_neigh,
                                       mode='connectivity', include_self=False)
    user_self_neighs = user_user_graph.tocoo().col
    user_self_neighs = np.array(np.array_split(user_self_neighs, num_user)).tolist()

    item_item_graph = kneighbors_graph(train_matrix.transpose(), num_self_neigh,
                                       mode='connectivity', include_self=False)
    item_self_neighs = item_item_graph.tocoo().col
    item_self_neighs = np.array(np.array_split(item_self_neighs, num_item)).tolist()

    assert len(train_set) == len(user_self_neighs)

    user_distances = build_distance_mp_map(train_set, u_adj_list, v_adj_list, num_workers=10, tag='user', norm=True)
    user_distances = flatten(user_distances, u_adj_list,
                             thre=10)

    item_start_id = num_user
    user_file = out_path + "/user.csv"
    item_file = out_path + "/item.csv"
    train_file = out_path + "/rating_train.csv"
    test_file = out_path + "/rating_test.csv"
    with open(user_file, 'a+') as user_f:
        for k in user_inverse_mapping:
            print(k + ',' + str(user_inverse_mapping[k]), file=user_f)
    with open(item_file, 'a+') as item_f:
        for k in item_inverse_mapping:
            print(k + ',' + str(item_inverse_mapping[k] + item_start_id), file=item_f)
    with open(train_file, 'a+') as train_f:
        print("src_id,dst_id,type", file=train_f)
        for user in u_adj_list:
            for item in sorted(list(u_adj_list[user])):
                print(str(user) + ',' + str(item + item_start_id) + ',0', file=train_f)
        for item in v_adj_list:
            for user in v_adj_list[item]:
                print(str(item + item_start_id) + ',' + str(user) + ',1', file=train_f)
        src_user = 0
        for users in user_self_neighs:
            for dst_user in users:
                print(str(src_user) + ',' + str(dst_user) + ',2', file=train_f)
            src_user += 1
        src_item = 0
        for items in item_self_neighs:
            for dst_item in items:
                print(str(src_item + item_start_id) + ',' + str(dst_item + item_start_id) + ',3', file=train_f)
            src_item += 1
    with open(test_file, 'a+') as test_f:
        print("src_id,dst_id,type", file=test_f)
        user = 0
        for items in test_set:
            for item in items:
                print(str(user) + ',' + str(item + item_start_id) + ',0', file=test_f)
            user += 1
        user = 0
        for items in test_set:
            for item in items:
                print(str(item + item_start_id) + ',' + str(user) + ',1', file=test_f)
            user += 1

    print('start generating sampled graphs...')
    num_graphs = 5
    for i in range(num_graphs):
        print('=== info: sampling graph {} / {}'.format(i + 1, num_graphs))
        sampled_user_graph = sample_graph_copying(node_neighbors_dict=u_adj_list,
                                                  distances=user_distances, epsilon=0.01)

        print('avg. sampled user-item graph degree: ',
              np.mean([len(x) for x in [*sampled_user_graph.values()]]))

        sampled_item_graph = {x: set() for x in range(num_item)}

        for k, items in sampled_user_graph.items():
            for x in items:
                sampled_item_graph[x].add(k)

        print('avg. sampled item-user graph degree: ',
              np.mean([len(x) for x in [*sampled_item_graph.values()]]))

        sampled_file = out_path + "/rating_sampled" + str(i) + ".csv"
        with open(sampled_file, 'a+') as sampled_f:
            print("src_id,dst_id,type", file=sampled_f)
            for user in sampled_user_graph:
                for item in sampled_user_graph[user]:
                    print(str(user) + ',' + str(item + item_start_id) + ',0', file=sampled_f)
            for item in sampled_item_graph:
                for user in sampled_item_graph[item]:
                    print(str(item + item_start_id) + ',' + str(user) + ',1', file=sampled_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Converting Data')
    parser.add_argument('--src_path', type=str, default="/tmp/",
                        help='source data directory')
    parser.add_argument('--out_path', type=str, default="/tmp/",
                        help='output directory')
    args = parser.parse_args()

    trans(args.src_path + "/", "Beauty", args.out_path + "/")
