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

"""assessment methods"""

import numpy as np


class Accuracy:
    """Accuracy"""
    def __init__(self):
        self.acc_num = 0
        self.total_num = 0
        self.name = 'Accuracy'

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        self.acc_num += np.sum(labels == logit_id)
        self.total_num += len(labels)

    def get_metrics(self):
        return self.acc_num / self.total_num * 100.0


class F1:
    """F1"""
    def __init__(self):
        self.logits_array = np.array([])
        self.labels_array = np.array([])
        self.name = 'F1'

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.argmax(logits, axis=1)
        self.labels_array = np.concatenate([self.labels_array, labels]).astype(np.bool)
        self.logits_array = np.concatenate([self.logits_array, logits]).astype(np.bool)

    def get_metrics(self):
        if len(self.labels_array) < 2:
            return 0.0
        tp = np.sum(self.labels_array & self.logits_array)
        fp = np.sum(self.labels_array & (~self.logits_array))
        fn = np.sum((~self.labels_array) & self.logits_array)
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        return 2.0 * p * r / (p + r) * 100.0


class Pearsonr:
    """Pearsonr"""
    def __init__(self):
        self.logits_array = np.array([])
        self.labels_array = np.array([])
        self.name = 'Pearsonr'

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.reshape(logits, -1)
        self.labels_array = np.concatenate([self.labels_array, labels])
        self.logits_array = np.concatenate([self.logits_array, logits])

    def get_metrics(self):
        if len(self.labels_array) < 2:
            return 0.0
        x_mean = self.logits_array.mean()
        y_mean = self.labels_array.mean()
        xm = self.logits_array - x_mean
        ym = self.labels_array - y_mean
        norm_xm = np.linalg.norm(xm)
        norm_ym = np.linalg.norm(ym)
        return np.dot(xm / norm_xm, ym / norm_ym) * 100.0


class Matthews:
    """Matthews"""
    def __init__(self):
        self.logits_array = np.array([])
        self.labels_array = np.array([])
        self.name = 'Matthews'

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.argmax(logits, axis=1)
        self.labels_array = np.concatenate([self.labels_array, labels]).astype(np.bool)
        self.logits_array = np.concatenate([self.logits_array, logits]).astype(np.bool)

    def get_metrics(self):
        if len(self.labels_array) < 2:
            return 0.0
        tp = np.sum(self.labels_array & self.logits_array)
        fp = np.sum(self.labels_array & (~self.logits_array))
        fn = np.sum((~self.labels_array) & self.logits_array)
        tn = np.sum((~self.labels_array) & (~self.logits_array))
        return (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) * 100.0
