# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Inference result aggregator."""

import copy


class Recommendation:
    """Recommendation."""

    class Path:
        """Item path."""
        def __init__(self, relation1, entity, relation2, hist_item, importance):
            self.relation1 = relation1
            self.entity = entity
            self.relation2 = relation2
            self.hist_item = hist_item
            self.importance = importance

    class ItemRecord:
        """Recommended item info."""
        def __init__(self, item, score):
            self.item = item
            self.score = score
            # paths must be sorted with importance in descending order
            self.paths = []

    def __init__(self, user):
        self.user = user
        # item_records must be sorted with score in descending order
        self.item_records = []


class InferenceAggregator:
    """
    Inference result aggregator.

    Args:
        top_k (int): The number of items to be recommended for each distinct user.
    """
    def __init__(self, top_k=1):
        if top_k < 1:
            raise ValueError('top_k is less than 1.')
        self._top_k = top_k
        self._user_recomms = dict()
        self._paths_sorted = False

    def aggregate(self, user, item, relation1, entity, relation2, hist_item, item_score, path_importance):
        """
        Aggregate inference results.

        Args:
            user (Tensor): User IDs, int Tensor in shape of [N, ].
            item (Tensor): Candidate item IDs, int Tensor in shape of [N, ].
            relation1 (Tensor): IDs of item-entity relations, int Tensor in shape of [N, <no. of per-item path>].
            entity (Tensor): Entity IDs, int Tensor in shape of [N, <no. of per-item path>].
            relation2 (Tensor): IDs of entity-hist_item relations, int Tensor in shape of [N, <no. of per-item path>].
            hist_item (Tensor): Historical item IDs, int Tensor in shape of [N, <no. of per-item path>].
            item_score (Tensor): TBNet output, recommendation scores of candidate items, float Tensor in shape of [N, ].
            path_importance (Tensor): TBNet output, the importance of each item to hist_item path for the
                recommendations, float Tensor in shape of [N, <no. of per-item path>].
        """
        user = user.asnumpy()
        item = item.asnumpy()
        relation1 = relation1.asnumpy()
        entity = entity.asnumpy()
        relation2 = relation2.asnumpy()
        hist_item = hist_item.asnumpy()
        item_score = item_score.asnumpy()
        path_importance = path_importance.asnumpy()

        batch_size = user.shape[0]

        added_users = set()
        for i in range(batch_size):
            if self._add(user[i], item[i], relation1[i], entity[i], relation2[i],
                         hist_item[i], item_score[i], path_importance[i]):
                added_users.add(user[i])
                self._paths_sorted = False

        for added_user in added_users:
            recomm = self._user_recomms[added_user]
            if len(recomm.item_records) > self._top_k:
                recomm.item_records = recomm.item_records[0:self._top_k]

    def recommend(self):
        """
        Generate recommendations for all distinct users.

        Returns:
            dict[int, Recommendation], a dictionary with user id as keys and Recommendation objects as values.
        """
        if not self._paths_sorted:
            self._sort_paths()
        return copy.deepcopy(self._user_recomms)

    def _add(self, user, item, relation1, entity, relation2, hist_item, item_score, path_importance):
        """Add a single infer record."""

        recomm = self._user_recomms.get(user, None)
        if recomm is None:
            recomm = Recommendation(user)
            self._user_recomms[user] = recomm

        # insert at the appropriate position
        for i, old_item_rec in enumerate(recomm.item_records):
            if i >= self._top_k:
                return False
            if item_score > old_item_rec.score:
                rec = self._infer_2_item_rec(item, relation1, entity, relation2,
                                             hist_item, item_score, path_importance)
                recomm.item_records.insert(i, rec)
                return True

        # append if has rooms
        if len(recomm.item_records) < self._top_k:
            rec = self._infer_2_item_rec(item, relation1, entity, relation2,
                                         hist_item, item_score, path_importance)
            recomm.item_records.append(rec)
            return True

        return False

    @staticmethod
    def _infer_2_item_rec(item, relation1, entity, relation2, hist_item, item_score, path_importance):
        """Converts a single infer result to a item record."""
        item_rec = Recommendation.ItemRecord(item, item_score)
        num_paths = path_importance.shape[0]
        for i in range(num_paths):
            path = Recommendation.Path(relation1[i], entity[i], relation2[i], hist_item[i], path_importance[i])
            item_rec.paths.append(path)
        return item_rec

    def _sort_paths(self):
        """Sort all item paths."""
        for recomm in self._user_recomms.values():
            for item_rec in recomm.item_records:
                item_rec.paths.sort(key=lambda x: x.importance, reverse=True)
        self._paths_sorted = True
