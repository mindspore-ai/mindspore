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

"""assessment methods"""

import numpy as np

class Accuracy():
    """Accuracy"""
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)

class F1():
    """F1"""
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, logits, labels):
        """Update F1 score"""
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        logit_id = np.reshape(logit_id, -1)
        pos_eva = np.isin(logit_id, [2, 3, 4, 5, 6, 7])
        pos_label = np.isin(labels, [2, 3, 4, 5, 6, 7])
        self.TP += np.sum(pos_eva & pos_label)
        self.FP += np.sum(pos_eva & (~pos_label))
        self.FN += np.sum((~pos_eva) & pos_label)
        print("-----------------precision is ", self.TP / (self.TP + self.FP))
        print("-----------------recall is ", self.TP / (self.TP + self.FN))
