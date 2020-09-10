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
Recommendation metrics
"""
import math
import heapq
from multiprocessing import Pool

import numpy as np

from src.utils import convert_item_id


def ndcg_k(actual, predicted, topk):
    """Calculates the normalized discounted cumulative gain at k"""
    idcg = idcg_k(actual, topk)
    res = 0

    dcg_k = sum([int(predicted[j] in set(actual)) / math.log(j + 2, 2) for j in range(topk)])
    res += dcg_k / idcg
    return res


def idcg_k(actual, k):
    """Calculates the ideal discounted cumulative gain at k"""
    res = sum([1.0 / math.log(i + 2, 2) for i in range(min(k, len(actual)))])
    return 1.0 if not res else res


def recall_at_k_2(r, k, all_pos_num):
    """Calculates the recall at k"""
    r = np.asfarray(r)[:k]
    return np.sum(r) / all_pos_num


def novelty_at_k(topk_items, item_degree_dict, num_user, k):
    """Calculate the novelty at k"""
    avg_nov = []
    for item in topk_items[:k]:
        avg_nov.append(-np.log2((item_degree_dict[item] + 1e-8) / num_user))
    return np.mean(avg_nov)


def ranklist_by_heapq(user_pos_test, test_items, rating, Ks):
    """Return the n largest score from the item_score by heap algorithm"""
    item_score = {}
    for i in test_items:
        item_score[i] = rating[i]

    K_max = max(Ks)
    K_max_item_score = heapq.nlargest(K_max, item_score, key=item_score.get)

    r = []
    for i in K_max_item_score:
        if i in user_pos_test:
            r.append(1)
        else:
            r.append(0)
    return r, K_max_item_score


def get_performance(user_pos_test, r, K_max_item, item_degree_dict, num_user, Ks):
    """Wraps the model metrics"""
    recall, ndcg, novelty = [], [], []
    for K in Ks:
        recall.append(recall_at_k_2(r, K, len(user_pos_test)))
        ndcg.append(ndcg_k(user_pos_test, K_max_item, K))
        novelty.append(novelty_at_k(K_max_item, item_degree_dict, num_user, K))
    return {'recall': np.array(recall), 'ndcg': np.array(ndcg), 'nov': np.array(novelty)}


class BGCFEvaluate:
    """
        Evaluate the model recommendation performance
    """

    def __init__(self, parser, train_graph, test_graph, Ks):
        self.num_user = train_graph.graph_info()["node_num"][0]
        self.num_item = train_graph.graph_info()["node_num"][1]
        self.Ks = Ks

        self.test_set = []
        self.train_set = []
        for i in range(0, self.num_user):
            train_item = train_graph.get_all_neighbors(node_list=[i], neighbor_type=1)
            train_item = train_item[1:]
            self.train_set.append(train_item)
        for i in range(0, self.num_user):
            test_item = test_graph.get_all_neighbors(node_list=[i], neighbor_type=1)
            test_item = test_item[1:]
            self.test_set.append(test_item)
        self.train_set = convert_item_id(self.train_set, self.num_user).tolist()
        self.test_set = convert_item_id(self.test_set, self.num_user).tolist()

        self.item_deg_dict = {}
        self.item_full_set = []
        for i in range(self.num_user, self.num_user + self.num_item):
            train_users = train_graph.get_all_neighbors(node_list=[i], neighbor_type=0)
            train_users = train_users.tolist()
            if isinstance(train_users, int):
                train_users = []
            else:
                train_users = train_users[1:]
            self.item_deg_dict[i - self.num_user] = len(train_users)
            test_users = test_graph.get_all_neighbors(node_list=[i], neighbor_type=0)
            test_users = test_users.tolist()
            if isinstance(test_users, int):
                test_users = []
            else:
                test_users = test_users[1:]
            self.item_full_set.append(train_users + test_users)

    def test_one_user(self, x):
        """Calculate one user metrics"""
        rating = x[0]
        u = x[1]

        training_items = self.train_set[u]

        user_pos_test = self.test_set[u]

        all_items = set(range(self.num_item))

        test_items = list(all_items - set(training_items))

        r, k_max_items = ranklist_by_heapq(user_pos_test, test_items, rating, self.Ks)

        return get_performance(user_pos_test, r, k_max_items, self.item_deg_dict, self.num_user, self.Ks), \
               [k_max_items[:self.Ks[x]] for x in range(len(self.Ks))]

    def eval_with_rep(self, user_rep, item_rep, parser):
        """Evaluation with user and item rep"""
        result = {'recall': np.zeros(len(self.Ks)), 'ndcg': np.zeros(len(self.Ks)),
                  'nov': np.zeros(len(self.Ks))}
        pool = Pool(parser.workers)
        user_indexes = np.arange(self.num_user)

        rating_preds = user_rep @ item_rep.transpose()
        user_rating_uid = zip(rating_preds, user_indexes)
        all_result = pool.map(self.test_one_user, user_rating_uid)

        top20 = []

        for re in all_result:
            result['recall'] += re[0]['recall'] / self.num_user
            result['ndcg'] += re[0]['ndcg'] / self.num_user
            result['nov'] += re[0]['nov'] / self.num_user
            top20.append(re[1][2])

        pool.close()

        sedp = [[] for i in range(len(self.Ks) - 1)]

        num_all_links = np.sum([len(x) for x in self.item_full_set])

        for k in range(len(self.Ks) - 1):
            for u in range(self.num_user):
                diff = []
                pred_items_at_k = all_result[u][1][k]
                for item in pred_items_at_k:
                    if item in self.test_set[u]:
                        avg_prob_all_user = len(self.item_full_set[item]) / num_all_links
                        diff.append(max((self.Ks[k] - pred_items_at_k.index(item) - 1)
                                        / (self.Ks[k] - 1) - avg_prob_all_user, 0))
                one_user_sedp = sum(diff) / self.Ks[k]
                sedp[k].append(one_user_sedp)

        sedp = np.array(sedp).mean(1)

        return result['recall'].tolist(), result['ndcg'].tolist(), \
               [sedp[1], sedp[2]], result['nov'].tolist()
