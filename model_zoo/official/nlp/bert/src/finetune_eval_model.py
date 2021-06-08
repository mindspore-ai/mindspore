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
Bert finetune and evaluation model script.
'''

import mindspore.nn as nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.ops import operations as P
from .bert_model import BertModel

class BertCLSModel(nn.Cell):
    """
    This class is responsible for classification task evaluation, i.e. XNLI(num_labels=3),
    LCQMC(num_labels=2), Chnsenti(num_labels=2). The returned output represents the final
    logits as the results of log_softmax is proportional to that of softmax.
    """
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLSModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.assessment_method = assessment_method

    def construct(self, input_ids, input_mask, token_type_id):
        _, pooled_output, _ = \
            self.bert(input_ids, token_type_id, input_mask)
        cls = self.cast(pooled_output, self.dtype)
        cls = self.dropout(cls)
        logits = self.dense_1(cls)
        logits = self.cast(logits, self.dtype)
        if self.assessment_method != "spearman_correlation":
            logits = self.log_softmax(logits)
        return logits

class BertSquadModel(nn.Cell):
    '''
    This class is responsible for SQuAD
    '''
    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquadModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.dense1 = nn.Dense(config.hidden_size, num_labels, weight_init=self.weight_init,
                               has_bias=True).to_float(config.compute_type)
        self.num_labels = num_labels
        self.dtype = config.dtype
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.is_training = is_training

    def construct(self, input_ids, input_mask, token_type_id):
        sequence_output, _, _ = self.bert(input_ids, token_type_id, input_mask)
        batch_size, seq_length, hidden_size = P.Shape()(sequence_output)
        sequence = P.Reshape()(sequence_output, (-1, hidden_size))
        logits = self.dense1(sequence)
        logits = P.Cast()(logits, self.dtype)
        logits = P.Reshape()(logits, (batch_size, seq_length, self.num_labels))
        logits = P.Transpose()(logits, (0, 2, 1))
        logits_shape = P.Shape()(logits)
        logits = self.log_softmax(P.Reshape()(logits, (-1, logits_shape[2])))
        logits = P.Transpose()(P.Reshape()(logits, logits_shape), (0, 2, 1))
        return logits

class BertNERModel(nn.Cell):
    """
    This class is responsible for sequence labeling task evaluation, i.e. NER(num_labels=11).
    The returned output represents the final logits as the results of log_softmax is proportional to that of softmax.
    """
    def __init__(self, config, is_training, num_labels=11, use_crf=False, dropout_prob=0.0,
                 use_one_hot_embeddings=False):
        super(BertNERModel, self).__init__()
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.hidden_probs_dropout_prob = 0.0
        self.bert = BertModel(config, is_training, use_one_hot_embeddings)
        self.cast = P.Cast()
        self.weight_init = TruncatedNormal(config.initializer_range)
        self.log_softmax = P.LogSoftmax(axis=-1)
        self.dtype = config.dtype
        self.num_labels = num_labels
        self.dense_1 = nn.Dense(config.hidden_size, self.num_labels, weight_init=self.weight_init,
                                has_bias=True).to_float(config.compute_type)
        self.dropout = nn.Dropout(1 - dropout_prob)
        self.reshape = P.Reshape()
        self.shape = (-1, config.hidden_size)
        self.use_crf = use_crf
        self.origin_shape = (-1, config.seq_length, self.num_labels)

    def construct(self, input_ids, input_mask, token_type_id):
        """Return the final logits as the results of log_softmax."""
        sequence_output, _, _ = \
            self.bert(input_ids, token_type_id, input_mask)
        seq = self.dropout(sequence_output)
        seq = self.reshape(seq, self.shape)
        logits = self.dense_1(seq)
        logits = self.cast(logits, self.dtype)
        if self.use_crf:
            return_value = self.reshape(logits, self.origin_shape)
        else:
            return_value = self.log_softmax(logits)
        return return_value
