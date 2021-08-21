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
"""KTNET train model"""

# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore.common.initializer import Normal, XavierUniform
from .layers import MemoryLayer, TriLinearTwoTimeSelfAttentionLayer
from .bert import BertConfig, BertModel


class KTNET(nn.Cell):
    """KTNET train model"""
    def __init__(self,
                 bert_config,  # paddle的bertconfig
                 max_wn_concept_length,
                 max_nell_concept_length,
                 wn_vocab_size,  # wn_concept_embedding_mat.shape[0]
                 wn_embedding_size,  # wn_concept_embedding_mat.shape[1]
                 nell_vocab_size,  # nell_concept_embedding_mat.shape[0],
                 nell_embedding_size,  # nell_concept_embedding_mat.shape[1]
                 bert_size,
                 is_training=False,
                 freeze=False,
                 ):
        super(KTNET, self).__init__()
        self.embedding_wn = nn.Embedding(vocab_size=wn_vocab_size,
                                         embedding_size=wn_embedding_size,
                                         embedding_table=XavierUniform(),
                                         dtype=mindspore.float32)
        self.embedding_nell = nn.Embedding(vocab_size=nell_vocab_size,
                                           embedding_size=nell_embedding_size,
                                           embedding_table=XavierUniform(),
                                           dtype=mindspore.float32)
        self.Sub = ops.Sub()
        self.ExpandDims = ops.ExpandDims()
        self.Less = ops.Less()
        self.Cast = ops.Cast()
        self.Add = ops.Add()
        self.Softmax = ops.Softmax()
        self.Mul = ops.Mul()
        self.Cast = ops.Cast()
        self.Concat = ops.Concat(axis=2)
        self.transpose = ops.Transpose()
        self.slice = ops.Slice()
        self.shape = ops.Shape()
        self.equal = ops.Equal()
        self.unstack = ops.Split(0, 2)
        self.mean = ops.ReduceMean()
        self.zeros = ops.Zeros()
        self.ones = ops.Ones()
        self.oneslike = ops.OnesLike()
        self.loss = nn.SoftmaxCrossEntropyWithLogits(sparse=True)
        self.dense1 = nn.Dense(in_channels=7488,  # att_output.shape[-1],
                               out_channels=2,
                               weight_init=Normal(sigma=0.02))
        # wn_concept_vocab_size = wn_concept_embedding_mat.shape[0]
        wn_concept_dim = wn_embedding_size  # wn_concept_embedding_mat.shape[1]
        # nell_concept_vocab_size = nell_concept_embedding_mat.shape[0]
        nell_concept_dim = nell_embedding_size  # nell_concept_embedding_mat.shape[1]
        self.reduce_sum = ops.ReduceSum()
        self.MemoryLayer_wn = MemoryLayer(bert_size, bert_config, max_wn_concept_length, wn_concept_dim,
                                          mem_method='raw', prefix='wn')
        self.MemoryLayer_nell = MemoryLayer(bert_size, bert_config, max_nell_concept_length, nell_concept_dim,
                                            mem_method='raw', prefix='nell')
        memory_output_size = bert_config['hidden_size'] + wn_concept_dim + nell_concept_dim
        self.TriLinearTwoTimeSelfAttentionLayer = TriLinearTwoTimeSelfAttentionLayer(memory_output_size,
                                                                                     dropout_rate=0.0,
                                                                                     cat_mul=True, cat_sub=True,
                                                                                     cat_twotime=True,
                                                                                     cat_twotime_mul=False,
                                                                                     cat_twotime_sub=True)

        self.bert_net_cfg_base = BertConfig(
            seq_length=384,
            vocab_size=28996,
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            intermediate_size=4096,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            use_relative_positions=False,
            dtype=mindspore.float32,
            compute_type=mindspore.float16
        )

        self.bert = BertModel(self.bert_net_cfg_base, True)

    def construct(self, input_mask, src_ids, pos_ids, sent_ids, wn_concept_ids, nell_concept_ids, start_positions,
                  end_positions):
        """
        1st Layer: BERT Layer
        2nd layer: Memory Layer
        3rd layer: Self-Matching Layer
        4th layer: Output Layer
        """
        bert = self.bert(src_ids, sent_ids, input_mask)

        enc_out = bert[0]  # return sequence_output
        # if freeze:#默认是false
        # enc_out.stop_gradient=True
        # logger.info("enc_out.stop_gradient: {}".format(false))#logger.info(
        # "enc_out.stop_gradient: {}".format(enc_out.stop_gradient))

        wn_memory_embs = self.embedding_wn(wn_concept_ids)
        nell_memory_embs = self.embedding_nell(nell_concept_ids)

        # get memory length
        wn_concept_ids_reduced = self.equal(wn_concept_ids,
                                            self.zeros(1,
                                                       mindspore.int32))  # [batch_size, sent_size, concept_size, 1]
        wn_concept_ids_reduced = self.Cast(wn_concept_ids_reduced,
                                           mindspore.float32)  # [batch_size, sent_size, concept_size, 1]
        wn_concept_ids_reduced = self.Mul(
            self.Sub(
                wn_concept_ids_reduced,
                self.ones(1, mindspore.float32)
            ), -1)

        wn_mem_length = self.reduce_sum(wn_concept_ids_reduced, 2)  # [batch_size, sent_size, 1]

        nell_concept_ids_reduced = self.equal(nell_concept_ids,
                                              self.zeros(1,
                                                         mindspore.int32))  # [batch_size, sent_size, concept_size, 1]
        nell_concept_ids_reduced = self.Cast(nell_concept_ids_reduced,
                                             mindspore.float32)  # [batch_size, sent_size, concept_size, 1]
        nell_concept_ids_reduced = self.Mul(
            self.Sub(
                nell_concept_ids_reduced,
                self.ones(1, mindspore.float32)
            ), -1)
        nell_mem_length = self.reduce_sum(nell_concept_ids_reduced, 2)  # [batch_size, sent_size, 1]

        # select and integrate
        wn_memory_embs = mnp.squeeze(wn_memory_embs, axis=3)
        wn_memory_output = self.MemoryLayer_wn(enc_out, wn_memory_embs, wn_mem_length, ignore_no_memory_token=True)

        nell_memory_embs = mnp.squeeze(nell_memory_embs, axis=3)
        nell_memory_output = self.MemoryLayer_nell(enc_out, nell_memory_embs, nell_mem_length,
                                                   ignore_no_memory_token=True)

        memory_output = self.Concat((enc_out, wn_memory_output, nell_memory_output))

        # do matching
        att_output = self.TriLinearTwoTimeSelfAttentionLayer(memory_output, input_mask)  # [bs, sq, concat_hs]

        logits = self.dense1(att_output)

        logits = self.transpose(logits, (2, 0, 1))
        start_logits, end_logits = self.unstack(logits)
        start_positions = mnp.squeeze(start_positions)
        end_positions = mnp.squeeze(end_positions)

        start_logits = mnp.squeeze(start_logits)
        start_loss = self.loss(start_logits, start_positions)
        start_loss = self.mean(start_loss)

        end_logits = mnp.squeeze(end_logits)
        end_loss = self.loss(end_logits, end_positions)
        end_loss = self.mean(end_loss)
        total_loss = (start_loss + end_loss) / 2.0

        return total_loss
