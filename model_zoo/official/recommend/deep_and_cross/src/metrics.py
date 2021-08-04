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

"""
Area under cure metric
"""

from sklearn.metrics import roc_auc_score
from mindspore import context
from mindspore.nn.metrics import Metric
from mindspore.communication.management import get_rank, get_group_size

class AUCMetric(Metric):
    """
    Area under cure metric
    """

    def __init__(self):
        super(AUCMetric, self).__init__()
        self.clear()
        self.full_batch = context.get_auto_parallel_context("full_batch")

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs): # inputs
        """Update list of predicts and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist() # predict
        all_label = inputs[2].asnumpy().flatten().tolist() # label
        self.pred_probs.extend(all_predict)
        if self.full_batch:
            rank_id = get_rank()
            group_size = get_group_size()
            gap = len(all_label) // group_size
            self.true_labels.extend(all_label[rank_id*gap: (rank_id+1)*gap])
        else:
            self.true_labels.extend(all_label)

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size is not equal to pred_probs.size()')

        auc = roc_auc_score(self.true_labels, self.pred_probs)
        print("====" * 20 + " auc_metric  end")
        print("====" * 20 + " auc: {}".format(auc))
        return auc
