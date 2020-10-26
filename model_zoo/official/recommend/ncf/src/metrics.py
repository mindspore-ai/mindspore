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
"""NCF metrics calculation"""
import numpy as np

from mindspore.nn.metrics import Metric

class NCFMetric(Metric):
    """NCF metrics method"""
    def __init__(self):
        super(NCFMetric, self).__init__()
        self.hr = []
        self.ndcg = []
        self.weights = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.hr = []
        self.ndcg = []
        self.weights = []

    def hit(self, gt_item, pred_items):
        if gt_item in pred_items:
            return 1
        return 0

    def ndcg_function(self, gt_item, pred_items):
        if gt_item in pred_items:
            index = pred_items.index(gt_item)
            return np.reciprocal(np.log2(index + 2))
        return 0

    def update(self, batch_indices, batch_items, metric_weights):
        """Update hr and ndcg"""
        batch_indices = batch_indices.asnumpy()  # (num_user, topk)
        batch_items = batch_items.asnumpy()  # (num_user, 100)
        metric_weights = metric_weights.asnumpy()  # (num_user,)
        num_user = batch_items.shape[0]
        for user in range(num_user):
            if metric_weights[user]:
                recommends = batch_items[user][batch_indices[user]].tolist()
                items = batch_items[user].tolist()[-1]
                self.hr.append(self.hit(items, recommends))
                self.ndcg.append(self.ndcg_function(items, recommends))

    def eval(self):
        return np.mean(self.hr), np.mean(self.ndcg)
