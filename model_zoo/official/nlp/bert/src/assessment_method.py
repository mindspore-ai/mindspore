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

'''
Bert evaluation assessment method script.
'''
import math
import numpy as np
from mindspore.nn.metrics import ConfusionMatrixMetric
from .CRF import postprocess

class Accuracy():
    '''
    calculate accuracy
    '''
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
    def __init__(self, use_crf=False, num_labels=2, mode="Binary"):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.use_crf = use_crf
        self.num_labels = num_labels
        self.mode = mode
        if self.mode.lower() not in ("binary", "multilabel"):
            raise ValueError("Assessment mode not supported, support: [Binary, MultiLabel]")
        if self.mode.lower() != "binary":
            self.metric = ConfusionMatrixMetric(skip_channel=False, metric_name=("f1 score"),
                                                calculation_method=False, decrease="mean")

    def update(self, logits, labels):
        '''
        update F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        if self.use_crf:
            backpointers, best_tag_id = logits
            best_path = postprocess(backpointers, best_tag_id)
            logit_id = []
            for ele in best_path:
                logit_id.extend(ele)
        else:
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
            self.metric.update(pred, target)

    def eval(self):
        return self.metric.eval()


class MCC():
    '''
    Calculate Matthews Correlation Coefficient
    '''
    def __init__(self):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0
    def update(self, logits, labels):
        '''
        MCC update
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        labels = labels.astype(np.bool)
        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        logit_id = np.reshape(logit_id, -1)
        logit_id = logit_id.astype(np.bool)
        ornot = logit_id ^ labels

        self.TP += (~ornot & labels).sum()
        self.FP += (ornot & ~labels).sum()
        self.FN += (ornot & labels).sum()
        self.TN += (~ornot & ~labels).sum()

    def cal(self):
        mcc = (self.TP*self.TN - self.FP*self.FN)/math.sqrt((self.TP+self.FP)*(self.TP+self.FN) *
                                                            (self.TN+self.FP)*(self.TN+self.FN))
        return mcc

class Spearman_Correlation():
    '''
    Calculate Spearman Correlation Coefficient
    '''
    def __init__(self):
        self.label = []
        self.logit = []

    def update(self, logits, labels):
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)
        logits = logits.asnumpy()
        logits = np.reshape(logits, -1)
        self.label.append(labels)
        self.logit.append(logits)

    def cal(self):
        '''
        Calculate Spearman Correlation
        '''
        label = np.concatenate(self.label)
        logit = np.concatenate(self.logit)
        sort_label = label.argsort()[::-1]
        sort_logit = logit.argsort()[::-1]
        n = len(label)
        d_acc = 0
        for i in range(n):
            d = np.where(sort_label == i)[0] - np.where(sort_logit == i)[0]
            d_acc += d**2
        ps = 1 - 6*d_acc/n/(n**2-1)
        return ps
