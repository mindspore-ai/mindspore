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

'''
Ernie evaluation assessment method script.
'''
import numpy as np
from mindspore.nn.metrics import ConfusionMatrixMetric

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
    def __init__(self, num_labels=2, mode="Binary"):
        self.TP = 0
        self.FP = 0
        self.FN = 0
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

class SpanF1():
    '''
    calculate F1、precision and recall score in span manner for NER
    '''
    def __init__(self, label2id=None):
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.label2id = label2id
        if label2id is None:
            raise ValueError("label2id info should not be empty")
        self.id2label = {}
        for key, value in label2id.items():
            self.id2label[value] = key

    def tag2span(self, ids):
        '''
        ids list to span mode
        '''
        labels = np.array([self.id2label[id] for id in ids])
        spans = []
        prev_label = None
        for idx, tag in enumerate(labels):
            tag = tag.lower()
            cur_label, label = tag[:1], tag[2:]
            if cur_label == 'b':
                spans.append((label, [idx, idx]))
            elif cur_label == 'i' and prev_label in ('b', 'i') and label == spans[-1][0]:
                spans[-1][1][1] = idx
            elif cur_label == 'o':
                pass
            else:
                spans.append((label, [idx, idx]))
            prev_label = cur_label
        return [(span[0], (span[1][0], span[1][1] + 1)) for span in spans]


    def update(self, logits, labels):
        '''
        update span F1 score
        '''
        labels = labels.asnumpy()
        labels = np.reshape(labels, -1)

        logits = logits.asnumpy()
        logit_id = np.argmax(logits, axis=-1)
        logit_id = np.reshape(logit_id, -1)

        label_spans = self.tag2span(labels)
        pred_spans = self.tag2span(logit_id)
        for span in pred_spans:
            if span in label_spans:
                self.TP += 1
                label_spans.remove(span)
            else:
                self.FP += 1
        for span in label_spans:
            self.FN += 1
