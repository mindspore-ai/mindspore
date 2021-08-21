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
"""bert model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import mindspore
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as mnp
from mindspore import Parameter
from mindspore.common.initializer import initializer, XavierUniform, Constant, Normal


class MemoryLayer(nn.Cell):
    """Memory Layer"""
    def __init__(self, bert_size, bert_config, concept_size, mem_emb_size, mem_method='cat', prefix=None):
        super(MemoryLayer, self).__init__()
        self.initializer_range = bert_config['initializer_range']
        self.bert_size = bert_config['hidden_size']
        self.concept_size = concept_size
        self.mem_emb_size = mem_emb_size
        assert mem_method in ['add', 'cat', 'raw']
        self.mem_method = mem_method
        self.prefix = prefix

        self.dense_1 = nn.Dense(in_channels=bert_size, out_channels=mem_emb_size,
                                weight_init=Normal(self.initializer_range), has_bias=False)
        self.expandDims = ops.ExpandDims()
        self.less = ops.Less()
        self.zeros = ops.Zeros()
        self.cast = ops.Cast()
        self.mul = ops.Mul()
        self.add = ops.Add()
        self.softMax = ops.Softmax()
        self.matMul = nn.MatMul().to_float(mindspore.float16)
        self.dense_2 = nn.Dense(in_channels=mem_emb_size, out_channels=bert_size,
                                weight_init=XavierUniform(gain=1))
        self.concat = ops.Concat(axis=2)
        self.slice = ops.Slice()
        self.transpose = ops.Transpose()
        self.zerosLike = ops.ZerosLike()

        self.sentinel = Parameter(initializer(Constant(0),
                                              shape=mem_emb_size, dtype=mindspore.float32))
        self.concept_ordinal = Parameter(np.arange(0, 1 + concept_size).astype(np.float32))

    def construct(self, bert_output, memory_embs, mem_length, ignore_no_memory_token=True):
        """
        :param bert_output: [batch_size, seq_size, bert_size]
        :param memory_embs: [batch_size, seq_size, concept_size, mem_emb_size]
        :param mem_length: [batch_size, sent_size, 1]
        :param ignore_no_memory_token
        :return:
        """

        concept_size = self.concept_size

        projected_bert = self.dense_1(bert_output)  # [batch_size *seq_size, mem_emb_size]

        expanded_bert = self.expandDims(projected_bert, 2)  # [batch_size, seq_size, 1, mem_emb_size]

        memory_embs_squeeze = self.slice(memory_embs, (0, 0, 0, 0),
                                         (memory_embs.shape[0], memory_embs.shape[1], 1,
                                          memory_embs.shape[3]))  # [bs,sq,1,ms]
        memory_embs_zero = self.zerosLike(memory_embs_squeeze)
        sentinel = self.add(memory_embs_zero, self.sentinel)  # [bs,sq,1,ms]
        extended_memory = self.concat((sentinel, memory_embs))  # [batch_size, seq_size, 1+concept_size, mem_emb_size]
        extended_memory = self.transpose(extended_memory, (0, 1, 3, 2))
        # [batch_size, seq_size, mem_emb_size, 1+concept_size]
        memory_score = self.matMul(expanded_bert, extended_memory)
        memory_score = mnp.squeeze(memory_score, axis=[2])
        extended_memory = self.transpose(extended_memory, (0, 1, 3, 2))
        # extended_memory: [batch_size, seq_size, 1+concept_size, mem_emb_size]
        # memory_score: [batch_size, seq_size, 1+concept_size]

        memory_embs_zero = self.zerosLike(memory_score)
        concept_ordinal = self.add(memory_embs_zero, self.concept_ordinal)  # [bs,sq,1+cs]

        memory_reverse_mask = self.less(
            ops.repeat_elements(mem_length, rep=1 + concept_size, axis=2),
            concept_ordinal)
        # [batch_size, seq_size, 1+concept_size]
        memory_reverse_mask = self.cast(memory_reverse_mask, mindspore.float32)

        memory_reverse_masked_infinity = self.mul(memory_reverse_mask, -1e6)
        # [batch_size, seq_size, 1+concept_size]

        memory_score = self.add(memory_score, memory_reverse_masked_infinity)
        # [batch_size, seq_size, 1+concept_size]

        memory_att = self.softMax(memory_score)  # [batch_size, seq_size, 1+concept_size]
        memory_att = self.expandDims(memory_att, 2)  # [batch_size, seq_size, 1, 1+concept_size]

        summ = self.matMul(memory_att, extended_memory)  # [batch_size, seq_size, 1, mem_emb_size]
        summ = mnp.squeeze(summ, axis=[2])  # [batch_size, seq_size,mem_emb_size]

        if ignore_no_memory_token:
            memory_embs_zero = self.mul(mem_length, 0.0)
            smaller_tensor = self.add(memory_embs_zero, self.zeros(1, mindspore.float32))

            condition = self.less(smaller_tensor, mem_length)  # [bs, sq]
            summ = self.mul(summ, self.cast(condition, mindspore.float32))  # [bs, sq, ms]

        output = summ  # [batch_size, seq_size, mem_emb_size]

        return output


class TriLinearTwoTimeSelfAttentionLayer(nn.Cell):
    """Tri Linear Two Time Self Attention Layer"""
    def __init__(self, hidden_size, dropout_rate=0.0,
                 cat_mul=False, cat_sub=False, cat_twotime=False, cat_twotime_mul=False, cat_twotime_sub=False):
        super(TriLinearTwoTimeSelfAttentionLayer, self).__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.cat_mul = cat_mul
        self.cat_sub = cat_sub
        self.cat_twotime = cat_twotime
        self.cat_twotime_mul = cat_twotime_mul
        self.cat_twotime_sub = cat_twotime_sub

        self.Sub = ops.Sub()
        self.ExpandDims = ops.ExpandDims()
        self.Less = ops.Less()
        self.Cast = ops.Cast()
        self.Add = ops.Add()
        self.Softmax = ops.Softmax()
        self.MatMul = nn.MatMul().to_float(mindspore.float16)
        self.Mul = ops.Mul()
        self.Cast = ops.Cast()
        self.Concat = ops.Concat(axis=2)
        self.transpose = ops.Transpose()
        self.slice = ops.Slice()
        self.fill = ops.Fill()
        self.shape = ops.Shape()
        self.zeros = ops.Zeros()
        self.expand_dims = ops.ExpandDims()
        self.zerosLike = ops.ZerosLike()
        self.ones = ops.Ones()
        self.onesLike = ops.OnesLike()

        XavierInitializer_weight_mul = XavierUniform()
        XavierInitializer_weight_12 = XavierUniform()  # XavierInitializer_cqu(1, hidden_size,True)
        create_parameter_weight_mul = Parameter(
            initializer(XavierInitializer_weight_mul, [1, hidden_size], mindspore.float32))
        self.create_parameter_weight_mul = mnp.squeeze(create_parameter_weight_mul)
        create_parameter_weight_12 = Parameter(
            initializer(XavierInitializer_weight_12, [1, hidden_size], mindspore.float32))
        self.create_parameter_weight_12 = mnp.squeeze(create_parameter_weight_12)
        self.create_parameter_bias = Parameter(initializer(Constant(0), [1], mindspore.float32))

        # self.squeeze1 = np.squeeze(axis=1)

    def construct(self, hidden_emb, sequence_mask):
        """
        :param hidden_emb: [batch_size, seq_size, hidden_size]
        :param sequence_mask: [batch_size, seq_size, 1]
        :return:
        """

        bs_1_hs = self.slice(hidden_emb, (0, 0, 0), (hidden_emb.shape[0], 1, hidden_emb.shape[2]))  # [bs, 1, hs]

        weight_mul = self.create_parameter_weight_mul
        hidden_emb_transpose = self.transpose(hidden_emb, (0, 2, 1))  # [BS, HS, SQ] aji

        memory_embs_zero = self.zerosLike(hidden_emb)
        weight_mul = self.Add(memory_embs_zero, weight_mul)
        rmul_1 = self.Mul(hidden_emb, weight_mul)  # for "hidden * self.weight_mul". [bs, sq(i), hs(j)]
        rmul_2 = self.MatMul(rmul_1,
                             hidden_emb_transpose)  # [bs, sq(i), hs(j)] mul [bs, hs(j), sq(k)] = [bs, sq(i), sq(k)]

        weight_1 = self.create_parameter_weight_12  # hs

        memory_embs_zero = self.zerosLike(bs_1_hs)  # bs 1 hs
        weight_1 = self.Add(memory_embs_zero, weight_1)  # bs 1 hs
        weight_1 = self.transpose(weight_1, (0, 2, 1))  # bs hs 1
        r1 = self.MatMul(hidden_emb, weight_1)  # [BS, SQ, 1]  aik
        r1 = mnp.squeeze(r1, 2)  # [BS, SQ  aik
        dynamic_tensor = self.transpose(rmul_2, (1, 0, 2))  # [sq, bs, sq]
        memory_embs_zero = self.zerosLike(dynamic_tensor)  # [sq, bs, sq]
        r1 = self.Add(memory_embs_zero, r1)
        r1 = self.transpose(r1, (1, 2, 0))  # [bs, sq, sq(from 1)]

        weight_2 = self.create_parameter_weight_12  # hs
        memory_embs_zero = self.zerosLike(bs_1_hs)
        weight_2 = self.Add(memory_embs_zero, weight_2)
        r2 = self.MatMul(weight_2, hidden_emb_transpose)  # [BS, 1, SQ]  aki
        r2 = mnp.squeeze(r2, 1)  # [BS, SQ]  aik
        dynamic_tensor = self.transpose(rmul_2, (1, 0, 2))
        memory_embs_zero = self.zerosLike(dynamic_tensor)
        r2 = self.Add(memory_embs_zero, r2)
        r2 = self.transpose(r2, (1, 0, 2))  # [bs,sq(from 1),sq]

        bias = self.create_parameter_bias  # [BS, SQ, SQ]
        memory_embs_zero = self.zerosLike(rmul_2)
        bias = self.Add(memory_embs_zero, bias)

        sim_score = self.Add(r1, r2)
        sim_score = self.Add(sim_score, rmul_2)
        sim_score = self.Add(sim_score, bias)
        # [bs,sq,1]+[bs,1,sq]+[bs,sq,sq]+[bs,sq,sq]=[BS,SQ,SQ]

        sequence_mask = self.Cast(sequence_mask, mindspore.float32)  # [BS,SQ,1]
        softmax_mask = self.Sub(
            sequence_mask,
            self.ones(1, mindspore.float32))  # [BS,SQ,1]
        tensor_temp1 = self.onesLike(softmax_mask)
        tensor_temp0 = self.zerosLike(softmax_mask)
        tensor_temp = self.Sub(tensor_temp0, tensor_temp1)
        softmax_mask = self.Mul(softmax_mask, tensor_temp)

        softmax_mask = self.Mul(softmax_mask, -1e6)  # [BS,SQ,1]

        dynamic_tensor = self.transpose(sim_score, (1, 0, 2))
        memory_embs_zero = self.zerosLike(dynamic_tensor)
        softmax_mask = self.Add(memory_embs_zero, softmax_mask)
        softmax_mask = self.transpose(softmax_mask, (1, 0, 2))  # [BS,sq(1),SQ]
        sim_score = self.Add(sim_score, softmax_mask)  # [bs,sq,sq]+[bs,sq(1),sq]=[BS,SQ,SQ]

        attn_prob = self.Softmax(sim_score)  # [BS,SQ,SQ]
        weighted_sum = self.MatMul(attn_prob, hidden_emb)  # [bs,sq,sq]*[bs,sq,hs]=[BS,SQ,HS]
        weighted_sum = self.Cast(weighted_sum, mindspore.float32)

        out_tensors = self.Concat((hidden_emb, weighted_sum))  # [hidden_emb, weighted_sum]
        if self.cat_mul:
            out_tensors = self.Concat((out_tensors, self.Mul(hidden_emb, weighted_sum)))
        if self.cat_sub:
            out_tensors = self.Concat((out_tensors, self.Sub(hidden_emb, weighted_sum)))
        if self.cat_twotime:
            twotime_att_prob = self.MatMul(attn_prob, attn_prob)  # [bs,sq,sq]*[bs,sq,sq]=[BS,SQ,SQ]
            twotime_weited_sum = self.MatMul(twotime_att_prob, hidden_emb)  # [BS,SQ,HS]
            twotime_weited_sum = self.Cast(twotime_weited_sum, mindspore.float32)
            out_tensors = self.Concat((out_tensors, twotime_weited_sum))
        if self.cat_twotime_mul:
            twotime_att_prob = self.MatMul(attn_prob, attn_prob)  # [bs,sq,sq]*[bs,sq,sq]=[BS,SQ,SQ]
            twotime_weited_sum = self.MatMul(twotime_att_prob, hidden_emb)  # [BS,SQ,HS]
            out_tensors = self.Concat((out_tensors, self.Mul(hidden_emb, twotime_weited_sum)))
        if self.cat_twotime_sub:
            twotime_att_prob = self.MatMul(attn_prob, attn_prob)  # [bs,sq,sq]*[bs,sq,sq]=[BS,SQ,SQ]
            twotime_weited_sum = self.MatMul(twotime_att_prob, hidden_emb)  # [BS,SQ,HS]
            out_tensors = self.Concat((out_tensors, self.Sub(hidden_emb, twotime_weited_sum)))
        output = out_tensors  # self.Concat(out_tensors)  # [BS,SQ, HS+HS+....]

        return output
