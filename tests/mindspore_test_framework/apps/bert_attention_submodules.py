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

"""Bert submodules."""

# pylint: disable=missing-docstring, arguments-differ

import math
import numpy as np

import mindspore.common.dtype as mstype
import mindspore.ops.functional as F
from mindspore import nn
from mindspore.common.initializer import TruncatedNormal
from mindspore.common.tensor import Tensor
from mindspore.tests.models.Bert_NEZHA.bert_model import SaturateCast, RelaPosEmbeddingsGenerator
from mindspore.ops import operations as P


class BertAttentionQueryKeyMul(nn.Cell):
    def __init__(self,
                 batch_size,
                 from_tensor_width,
                 to_tensor_width,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 query_act=None,
                 key_act=None,
                 initializer_range=0.02):
        super(BertAttentionQueryKeyMul, self).__init__()
        self.from_tensor_width = from_tensor_width
        self.to_tensor_width = to_tensor_width
        self.units = num_attention_heads * size_per_head
        self.weight = TruncatedNormal(initializer_range)

        self.trans_shape = (0, 2, 1, 3)
        self.transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.shp_from_2d = (-1, self.from_tensor_width)
        self.shp_to_2d = (-1, self.to_tensor_width)
        self.query_layer = nn.Dense(self.from_tensor_width,
                                    self.units,
                                    activation=query_act,
                                    weight_init=self.weight)
        self.key_layer = nn.Dense(self.to_tensor_width,
                                  self.units,
                                  activation=key_act,
                                  weight_init=self.weight)

        self.shp_from = (batch_size, from_seq_length, num_attention_heads, size_per_head)
        self.shp_to = (
            batch_size, to_seq_length, num_attention_heads, size_per_head)

        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.cast = P.Cast()

    def construct(self, from_tensor, to_tensor):
        from_tensor_2d = self.reshape(from_tensor, self.shp_from_2d)
        to_tensor_2d = self.reshape(to_tensor, self.shp_to_2d)
        from_tensor_2d = self.cast(from_tensor_2d, mstype.float32)
        to_tensor_2d = self.cast(to_tensor_2d, mstype.float32)
        query_out = self.query_layer(from_tensor_2d)
        key_out = self.key_layer(to_tensor_2d)

        query_layer = self.reshape(query_out, self.shp_from)
        query_layer = self.transpose(query_layer, self.trans_shape)
        key_layer = self.reshape(key_out, self.shp_to)
        key_layer = self.transpose(key_layer, self.trans_shape)

        attention_scores = self.matmul_trans_b(query_layer, key_layer)

        return query_layer, key_layer, attention_scores


class BertAttentionRelativePositionKeys(nn.Cell):
    def __init__(self,
                 batch_size,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 use_relative_positions=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):
        super(BertAttentionRelativePositionKeys, self).__init__()
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.use_relative_positions = use_relative_positions
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.trans_shape_position = (1, 2, 0, 3)
        self.trans_shape_relative = (2, 0, 1, 3)

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))

        self.reshape = P.Reshape()
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.matmul_trans_b = P.BatchMatMul(transpose_b=True)
        self.batch_num = batch_size * num_attention_heads
        self.cast = P.Cast()

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        self._generate_relative_positions_embeddings = \
            RelaPosEmbeddingsGenerator(length=self.to_seq_length,
                                       depth=self.size_per_head,
                                       max_relative_position=16,
                                       initializer_range=initializer_range,
                                       use_one_hot_embeddings=use_one_hot_embeddings)

    def construct(self, input_tensor, query_layer):
        # use_relative_position, supplementary logic
        relations_keys_embeddings = self._generate_relative_positions_embeddings()
        if self.use_relative_positions:
            # 'relations_keys' = [F|T, F|T, H]
            relations_keys = self.cast_compute_type(relations_keys_embeddings)
            # query_layer_t is [F, B, N, H]
            query_layer_t = self.transpose(query_layer, self.trans_shape_relative)
            # query_layer_r is [F, B * N, H]
            query_layer_r = self.reshape(query_layer_t,
                                         (self.from_seq_length,
                                          self.batch_num,
                                          self.size_per_head))
            # key_position_scores is [F, B * N, F|T]
            query_layer_r = self.cast(query_layer_r, mstype.float32)
            key_position_scores = self.matmul_trans_b(query_layer_r,
                                                      relations_keys)
            # key_position_scores_r is [F, B, N, F|T]
            key_position_scores_r = self.reshape(key_position_scores,
                                                 (self.from_seq_length,
                                                  self.batch_size,
                                                  self.num_attention_heads,
                                                  self.from_seq_length))
            # key_position_scores_r_t is [B, N, F, F|T]
            key_position_scores_r_t = self.transpose(key_position_scores_r,
                                                     self.trans_shape_position)
            input_tensor = self.cast(input_tensor, mstype.float32)

            input_tensor = input_tensor + key_position_scores_r_t

        attention_scores = self.multiply(input_tensor, self.scores_mul)

        return relations_keys_embeddings, attention_scores


class BertAttentionMask(nn.Cell):
    def __init__(self,
                 has_attention_mask=False,
                 dtype=mstype.float32):

        super(BertAttentionMask, self).__init__()
        self.has_attention_mask = has_attention_mask
        self.multiply_data = Tensor([-1000.0,], dtype=dtype)
        self.multiply = P.Mul()

        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

    def construct(self, input_tensor, attention_mask):
        attention_scores = input_tensor
        attention_scores = self.cast(attention_scores, mstype.float32)
        if self.has_attention_mask:
            attention_mask = self.expand_dims(attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), mstype.float32),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)

        return attention_scores


class BertAttentionMaskBackward(nn.Cell):
    def __init__(self,
                 attention_mask_shape,
                 has_attention_mask=False,
                 dtype=mstype.float32):
        super(BertAttentionMaskBackward, self).__init__()
        self.has_attention_mask = has_attention_mask
        self.multiply_data = Tensor([-1000.0,], dtype=dtype)
        self.multiply = P.Mul()
        self.attention_mask = Tensor(np.ones(shape=attention_mask_shape).astype(np.float32))
        if self.has_attention_mask:
            self.expand_dims = P.ExpandDims()
            self.sub = P.Sub()
            self.add = P.Add()
            self.cast = P.Cast()
            self.get_dtype = P.DType()

    def construct(self, input_tensor):
        attention_scores = input_tensor
        attention_scores = self.cast(attention_scores, mstype.float32)
        if self.has_attention_mask:
            attention_mask = self.expand_dims(self.attention_mask, 1)
            multiply_out = self.sub(self.cast(F.tuple_to_array((1.0,)), mstype.float32),
                                    self.cast(attention_mask, self.get_dtype(attention_scores)))

            adder = self.multiply(multiply_out, self.multiply_data)
            attention_scores = self.add(adder, attention_scores)
        return attention_scores


class BertAttentionSoftmax(nn.Cell):
    def __init__(self,
                 batch_size,
                 to_tensor_width,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 value_act=None,
                 attention_probs_dropout_prob=0.0,
                 initializer_range=0.02):
        super(BertAttentionSoftmax, self).__init__()
        self.to_tensor_width = to_tensor_width
        self.value_act = value_act

        self.reshape = P.Reshape()

        self.shp_to_2d = (-1, self.to_tensor_width)
        self.shp_from = (batch_size, from_seq_length, num_attention_heads, size_per_head)
        self.shp_to = (
            batch_size, to_seq_length, num_attention_heads, size_per_head)

        self.trans_shape = (0, 2, 1, 3)
        self.trans_shape_start = (0, 1)
        self.matmul = P.BatchMatMul()

        self.units = num_attention_heads * size_per_head
        self.weight = TruncatedNormal(initializer_range)

        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=attention_probs_dropout_prob)
        self.transpose = P.Transpose()

        self.value_layer = nn.Dense(self.to_tensor_width,
                                    self.units,
                                    activation=value_act,
                                    weight_init=self.weight)
        self.cast = P.Cast()

    def construct(self, to_tensor, attention_scores):
        to_tensor = self.transpose(to_tensor, self.trans_shape_start)
        to_tensor_2d = self.reshape(to_tensor, self.shp_to_2d)
        to_tensor_2d = self.cast(to_tensor_2d, mstype.float32)
        value_out = self.value_layer(to_tensor_2d)

        attention_probs = self.softmax(attention_scores)
        attention_probs = self.cast(attention_probs, mstype.float32)

        value_layer = self.reshape(value_out, self.shp_to)
        value_layer = self.transpose(value_layer, self.trans_shape)

        context_layer = self.matmul(attention_probs, value_layer)

        return value_layer, context_layer


class BertAttentionRelativePositionValues(nn.Cell):
    def __init__(self,
                 batch_size,
                 from_seq_length,
                 to_seq_length,
                 num_attention_heads=1,
                 size_per_head=512,
                 use_one_hot_embeddings=False,
                 initializer_range=0.02,
                 do_return_2d_tensor=False,
                 use_relative_positions=False,
                 dtype=mstype.float32,
                 compute_type=mstype.float32):

        super(BertAttentionRelativePositionValues, self).__init__()
        self.batch_size = batch_size
        self.from_seq_length = from_seq_length
        self.to_seq_length = to_seq_length
        self.use_relative_positions = use_relative_positions
        self.size_per_head = size_per_head
        self.num_attention_heads = num_attention_heads
        self.trans_shape_position = (1, 2, 0, 3)
        self.trans_shape_relative = (2, 0, 1, 3)

        self.scores_mul = 1.0 / math.sqrt(float(self.size_per_head))
        self.trans_shape = (0, 2, 1, 3)

        self.reshape = P.Reshape()
        self.multiply = P.Mul()
        self.transpose = P.Transpose()
        self.batch_num = batch_size * num_attention_heads
        self.matmul = P.BatchMatMul()
        self.do_return_2d_tensor = do_return_2d_tensor
        if self.do_return_2d_tensor:
            self.shp_return = (batch_size * from_seq_length, num_attention_heads * size_per_head)
        else:
            self.shp_return = (batch_size, from_seq_length, num_attention_heads * size_per_head)

        self.cast_compute_type = SaturateCast(dst_type=compute_type)
        self._generate_relative_positions_embeddings = \
            RelaPosEmbeddingsGenerator(length=self.to_seq_length,
                                       depth=self.size_per_head,
                                       max_relative_position=16,
                                       initializer_range=initializer_range,
                                       use_one_hot_embeddings=use_one_hot_embeddings)
        self.fill = P.Fill()
        self.multiply = P.Mul()
        self.type = P.DType()
        self.cast = P.Cast()

    def construct(self, input_tensor, attention_probs):
        # use_relative_position, supplementary logic
        relations_values_embedding = self._generate_relative_positions_embeddings()  # (128, 128, 64)
        if self.use_relative_positions:
            # 'relations_values' = [F|T, F|T, H]
            relations_values = self.cast_compute_type(relations_values_embedding)
            # attention_probs_t is [F, B, N, T]
            attention_probs_t = self.transpose(attention_probs, self.trans_shape_relative)
            # attention_probs_r is [F, B * N, T]
            attention_probs_r = self.reshape(
                attention_probs_t,
                (self.from_seq_length,
                 self.batch_num,
                 self.to_seq_length))  # (128,768,128)
            # value_position_scores is [F, B * N, H]
            value_position_scores = self.matmul(attention_probs_r,
                                                relations_values)
            # value_position_scores_r is [F, B, N, H]
            value_position_scores_r = self.reshape(value_position_scores,
                                                   (self.from_seq_length,
                                                    self.batch_size,
                                                    self.num_attention_heads,
                                                    self.size_per_head))
            # value_position_scores_r_t is [B, N, F, H]
            value_position_scores_r_t = self.transpose(value_position_scores_r,
                                                       self.trans_shape_position)
            input_tensor = input_tensor + value_position_scores_r_t

        context_layer = self.transpose(input_tensor, self.trans_shape)
        context_layer = self.reshape(context_layer, self.shp_return)
        # ge reshape should not return, need an operator here
        ones = self.cast(self.fill((1, 1), 1), self.type(context_layer))
        context_layer = self.multiply(context_layer, ones)
        return relations_values_embedding, context_layer


class BertDense(nn.Cell):
    def __init__(self,
                 hidden_size=768,
                 intermediate_size=3072,
                 initializer_range=0.02):
        super(BertDense, self).__init__()
        self.intermediate = nn.Dense(in_channels=hidden_size,
                                     out_channels=intermediate_size,
                                     activation=None,
                                     weight_init=TruncatedNormal(
                                         initializer_range)
                                     )
        self.cast = P.Cast()

    def construct(self, attention_output):
        attention_output = self.cast(attention_output, mstype.float32)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output
