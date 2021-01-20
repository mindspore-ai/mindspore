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
    '''
    calculate F1 score
    '''
    def __init__(self, num_labels=2, mode="Binary"):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.num_labels = num_labels
        self.P = 0
        self.AP = 0
        self.mode = mode
        if self.mode.lower() not in ("binary", "multilabel"):
            raise ValueError("Assessment mode not supported, support: [Binary, MultiLabel]")

    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        logit_id = np.reshape(logit_id, -1)

        if self.mode.lower() == "binary":
            pos_eva = np.isin(logit_id, [i for i in range(1, self.num_labels)])
            pos_label = np.isin(labels, [i for i in range(1, self.num_labels)])
            self.TP += np.sum(pos_eva&pos_label)
            self.FP += np.sum(pos_eva&(~pos_label))
            self.FN += np.sum((~pos_eva)&pos_label)
        else:
            target = np.zeros((len(labels), self.num_labels), dtype=np.int)
            pred = np.zeros((len(logit_id), self.num_labels), dtype=np.int)
            for i, label in enumerate(labels):
                target[i][label] = 1
            for i, label in enumerate(logit_id):
                pred[i][label] = 1
            positives = pred.sum(axis=0)
            actual_positives = target.sum(axis=0)
            true_positives = (target * pred).sum(axis=0)
            self.TP += true_positives
            self.P += positives
            self.AP += actual_positives

    def eval(self):
        if self.mode.lower() == "binary":
            f1 = self.TP / (2 * self.TP + self.FP + self.FN)
        else:
            tp = np.sum(self.TP)
            p = np.sum(self.P)
            ap = np.sum(self.AP)
            f1 = 2 * tp / (ap + p)
        return f1
