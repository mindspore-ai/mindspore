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
"""TB-Net metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score
from mindspore.nn.metrics import Metric


class AUC(Metric):
    """TB-Net metrics method. Compute model metrics AUC."""

    def __init__(self):
        super(AUC, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs):
        """Update list of predictions and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist()
        all_label = inputs[2].asnumpy().flatten().tolist()
        self.pred_probs.extend(all_predict)
        self.true_labels.extend(all_label)

    def eval(self):
        """Return AUC score"""
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size is not equal to pred_probs.size()')

        auc = roc_auc_score(self.true_labels, self.pred_probs)

        return auc


class ACC(Metric):
    """TB-Net metrics method. Compute model metrics ACC."""

    def __init__(self):
        super(ACC, self).__init__()
        self.clear()

    def clear(self):
        """Clear the internal evaluation result."""
        self.true_labels = []
        self.pred_probs = []

    def update(self, *inputs):
        """Update list of predictions and labels."""
        all_predict = inputs[1].asnumpy().flatten().tolist()
        all_label = inputs[2].asnumpy().flatten().tolist()
        self.pred_probs.extend(all_predict)
        self.true_labels.extend(all_label)

    def eval(self):
        """Return accuracy score"""
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError(
                'true_labels.size is not equal to pred_probs.size()')

        predictions = [1 if i >= 0.5 else 0 for i in self.pred_probs]
        acc = np.mean(np.equal(predictions, self.true_labels))

        return acc
