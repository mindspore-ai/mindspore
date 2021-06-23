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
'''main model'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import mindspore.ops as P
import mindspore
import mindspore.nn as nn

from mindspore.common.initializer import initializer, TruncatedNormal
from src.bert_model import BertModel
from src.common.register import RegisterSet




class Dense(nn.Dense):
    def __init__(self, in_channels, out_channels, has_bias=True, activation=None, cfg=None):
        super().__init__(in_channels, out_channels, weight_init='normal', bias_init='zeros',
                         has_bias=has_bias, activation=activation)
        self.cfg = cfg
        self.reset_parameters()

    def reset_parameters(self):
        self.weight.set_data(initializer(TruncatedNormal(0.02), self.weight.shape))

class LossCell(nn.Cell):

    def __init__(self):
        super(LossCell, self).__init__()
        self.softmax = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        self.reshape = P.Reshape()
    def construct(self, fc_out, mask_label):
        mask_label = self.reshape(mask_label, (-1,))
        loss = self.softmax(fc_out, mask_label)
        return loss

class Model(nn.Cell):
    '''
    Main task model
    '''
    def __init__(self, model_params, is_training=True):
        super(Model, self).__init__()
        self.model_params = model_params
        self.is_training = is_training
        self.bert = BertModel(model_params, is_training)
        if self.is_training:
            self.dropout = nn.Dropout(keep_prob=0.9, dtype=mindspore.float32)
        else:
            self.dropout = nn.Dropout(keep_prob=1.0, dtype=mindspore.float32)
        self.dense = Dense(model_params.hidden_size, 2).to_float(mindspore.float16)
        self.cast = P.Cast()

    def construct(self, src_ids, sent_ids, input_mask):
        _, pooled_output, _ = self.bert(src_ids, sent_ids, input_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        logits = self.cast(logits, mindspore.float32)
        return logits

@RegisterSet.models.register
class RobertaOneSentClassificationEn(nn.Cell):
    '''
    model with loss
    '''
    def __init__(self, model_params, is_training=True):

        super(RobertaOneSentClassificationEn, self).__init__()
        self.bert = Model(model_params, is_training)
        self.loss = LossCell()
    def construct(self, tid, labels, input_mask, pos_ids, sent_ids, src_ids):
        logits = self.bert(src_ids, sent_ids, input_mask)
        loss = self.loss(logits, labels)
        return loss
