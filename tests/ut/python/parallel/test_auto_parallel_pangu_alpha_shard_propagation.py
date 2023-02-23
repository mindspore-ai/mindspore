# Copyright 2021-2022 Huawei Technologies Co., Ltd
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

import math
import re
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore.common.initializer import initializer
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.nn import Cell
from mindspore import context
from mindspore.nn.wrap.cell_wrapper import _VirtualDatasetCell
from mindspore.parallel import set_algo_parameters
from mindspore.nn.layer.activation import get_activation
from mindspore.train import Model
from mindspore.common.api import _cell_graph_executor
from tests.dataset_mock import MindData
context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Dataset(MindData):
    def __init__(self, *inputs, length=3):
        super(Dataset, self).__init__(size=length)
        self.inputs = inputs
        self.index = 0
        self.length = length

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= self.length:
            raise StopIteration
        self.index += 1
        return self.inputs

    def reset(self):
        self.index = 0


def set_parallel_configure_for_layer(network, layer_id, offset, layers):
    pp_dis = max(int((layers + 1) / 1), 1)
    pp_id = min((layer_id + offset) // pp_dis, 0)
    network.pipeline_stage = pp_id
    dis = max(int((layers + 1) / 4), 1)
    network.set_comm_fusion(int((layer_id + offset) / dis) + 1)


class _LayerNorm(Cell):
    def __init__(self, normalized_shape, eps=1e-5, param_init_type=mstype.float32):
        super(_LayerNorm, self).__init__()
        if normalized_shape[0] <= 1024:
            self.layer_norm = P.LayerNorm(begin_norm_axis=-1,
                                          begin_params_axis=-1,
                                          epsilon=eps)
        self.is_self_defined = normalized_shape[0] > 1024
        self.gamma = Parameter(initializer('ones', normalized_shape, param_init_type), name="gamma",
                               parallel_optimizer=False)
        self.beta = Parameter(initializer('zeros', normalized_shape, param_init_type), name="beta",
                              parallel_optimizer=False)
        self.mean = P.ReduceMean(keep_dims=True)
        self.square = P.Square()
        self.sqrt = P.Sqrt()
        self.sub1 = P.Sub()
        self.sub2 = P.Sub()
        self.add = P.Add()
        self.eps = eps
        self.mul = P.Mul()
        self.add2 = P.Add()
        self.real_div = P.RealDiv()

    def construct(self, x):
        if self.is_self_defined:
            mean = self.mean(x, -1)
            diff = self.sub1(x, mean)
            variance = self.mean(self.square(diff), -1)
            variance_eps = self.sqrt(self.add(variance, self.eps))
            output = self.real_div(diff, variance_eps)
            output = self.add2(self.mul(output, self.gamma), self.beta)
        else:
            output, _, _ = self.layer_norm(x, self.gamma, self.beta)
        return output

    def shard(self, strategy):
        if self.is_self_defined:
            self.mean.shard(strategy)
            self.square.shard(strategy)
            self.sqrt.shard(strategy)
            self.sub1.shard((strategy[0], strategy[0]))
            self.sub2.shard((strategy[0], strategy[0]))
            self.add.shard((strategy[0], ()))
            self.mul.shard((strategy[0], (1,)))
            self.add2.shard((strategy[0], (1,)))
            self.real_div.shard((strategy[0], strategy[0]))
        else:
            self.layer_norm.shard((strategy[0], (1,), (1,)))
        return self


class _Linear(Cell):
    def __init__(self,
                 in_channels,
                 out_channels,
                 weight_init='normal',
                 bias_init='zeros',
                 has_bias=True,
                 activation=None,
                 transpose_b=True,
                 param_init_type=mstype.float32,
                 compute_dtype=mstype.float16):
        super(_Linear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        weight_shape = [out_channels, in_channels] if transpose_b else [in_channels, out_channels]
        self.expert_flag = False
        self.weight = Parameter(initializer(weight_init, weight_shape, param_init_type), name="weight")
        self.matmul = P.MatMul(transpose_b=transpose_b)
        self.bias = None
        self.has_bias = has_bias
        if self.has_bias:
            self.bias = Parameter(initializer(bias_init, [out_channels], param_init_type), name="bias")
            self.bias_add = P.Add()
        self.act_name = activation
        self.activation = get_activation(activation) if isinstance(activation, str) else activation
        self.activation_flag = self.activation is not None
        self.dtype = compute_dtype
        self.cast = P.Cast()

    def construct(self, x):
        out_shape = P.Shape()(x)[:-1] + (self.out_channels,)
        x = P.Reshape()(x, (-1, self.in_channels))
        weight = self.cast(self.weight, self.dtype)
        x = self.matmul(x, weight)
        if self.has_bias:
            x = self.bias_add(x, self.cast(self.bias, self.dtype))
        if self.activation_flag:
            x = self.activation(x)
        output = P.Reshape()(x, out_shape)
        return output

    def shard(self, strategy_matmul, strategy_bias=None, strategy_activation=None):
        self.matmul.shard(strategy_matmul)
        if self.has_bias:
            self.bias_add.shard(strategy_bias)
        if self.activation_flag:
            if self.act_name.lower() == "leakyrelu":
                self.activation.select_op.shard((strategy_activation[0], strategy_activation[0]))
            elif self.act_name.lower() == "logsigmoid":
                self.activation.mul.shard((strategy_activation[0], ()))
                self.activation.exp.shard(strategy_activation)
                self.activation.add.shard((strategy_activation[0], ()))
                self.activation.rec.shard(strategy_activation)
                self.activation.log.shard(strategy_activation)
            else:
                getattr(self.activation, self.act_name).shard(strategy_activation)
        return self


class FeedForward(Cell):
    def __init__(self, hidden_size,
                 ffn_hidden_size,
                 dropout_rate,
                 hidden_act='gelu',
                 param_init_type=mstype.float32):
        super(FeedForward, self).__init__()
        dp = 2
        mp = 8
        input_size = hidden_size
        output_size = ffn_hidden_size
        self.mapping = _Linear(in_channels=input_size,
                               out_channels=output_size,
                               activation=hidden_act,
                               transpose_b=False,
                               param_init_type=param_init_type)
        self.projection = _Linear(in_channels=output_size,
                                  out_channels=input_size,
                                  transpose_b=False,
                                  param_init_type=param_init_type)
        self.projection.shard(strategy_matmul=((dp, mp), (mp, 1)))
        self.projection.bias.parallel_optimizer = False
        self.dropout = nn.Dropout(p=dropout_rate)
        self.dropout_3d = nn.Dropout(p=dropout_rate)
        self.cast = P.Cast()

    def construct(self, x):
        x = self.cast(x, mstype.float16)
        hidden = self.mapping(x)
        output = self.projection(hidden)
        if len(F.shape(output)) == 3:
            output = self.dropout_3d(output)
        else:
            output = self.dropout(output)
        return output


class MultiHeadAttention(Cell):
    def __init__(self, batch_size,
                 src_seq_length,
                 tgt_seq_length,
                 hidden_size,
                 num_heads,
                 hidden_dropout_rate=0.1,
                 attention_dropout_rate=0.1,
                 compute_dtype=mstype.float16,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32):
        super(MultiHeadAttention, self).__init__()
        self.src_seq_length = src_seq_length
        self.tgt_seq_length = tgt_seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.projection = _Linear(in_channels=hidden_size,
                                  out_channels=hidden_size,
                                  transpose_b=False,
                                  param_init_type=param_init_type).to_float(compute_dtype)
        self.projection.shard(strategy_matmul=((2, 8), (8, 1)))
        self.projection.bias.parallel_optimizer = False
        self.transpose = P.Transpose()
        self.merger_head_transpose = P.Transpose()
        self.reshape = P.Reshape()
        self.n_head = num_heads
        self.size_per_head = hidden_size // self.n_head
        self.concat_k = P.Concat(axis=3)
        self.concat_v = P.Concat(axis=2)
        self.multiply_data = Tensor([
            -10000.0,
        ], dtype=softmax_compute_type)
        self.batch_matmul = P.BatchMatMul()
        self.real_div = P.RealDiv()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.add = P.Add()
        self.scale_factor = Tensor(math.sqrt(self.size_per_head))
        self.dropout = nn.Dropout(p=hidden_dropout_rate)
        self.prob_dropout = nn.Dropout(p=attention_dropout_rate)
        self.softmax = nn.Softmax().to_float(softmax_compute_type)
        self.expand_dims = P.ExpandDims()
        # Query
        self.dense1 = _Linear(hidden_size,
                              hidden_size,
                              param_init_type=param_init_type).to_float(compute_dtype)
        # Key
        self.dense2 = _Linear(hidden_size,
                              hidden_size,
                              param_init_type=param_init_type).to_float(compute_dtype)
        # Value
        self.dense3 = _Linear(hidden_size,
                              hidden_size,
                              param_init_type=param_init_type).to_float(compute_dtype)
        self.dtype = compute_dtype
        self.softmax_dtype = softmax_compute_type

    def construct(self, query_tensor, key_tensor, value_tensor, attention_mask, key_past=None,
                  value_past=None, batch_valid_length=None):
        query_tensor, key_tensor, value_tensor, batch_size, ori_shape = self._convert_to_2d_tensor(query_tensor,
                                                                                                   key_tensor,
                                                                                                   value_tensor,
                                                                                                   attention_mask)
        query = self.dense1(query_tensor)
        key = self.dense2(key_tensor)
        value = self.dense3(value_tensor)
        query = self.transpose(
            F.reshape(
                query,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        key = self.transpose(
            F.reshape(
                key, (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 3, 1))
        value = self.transpose(
            F.reshape(
                value,
                (batch_size, -1, self.n_head, self.size_per_head)),
            (0, 2, 1, 3))
        if len(F.shape(attention_mask)) == 3:
            attention_mask = self.expand_dims(attention_mask, 1)
        key_present = key
        value_present = value
        layer_present = (key_present, value_present)
        attention = self._attn(query, key, value, attention_mask)
        output = self.projection(attention)
        output = self.dropout(output)
        output = F.reshape(output, ori_shape)
        return output, layer_present

    def _convert_to_2d_tensor(self, query_tensor, key_tensor, value_tensor, attention_mask):
        query_shape = F.shape(query_tensor)
        query_tensor = F.reshape(query_tensor, (-1, query_shape[-1]))
        key_shape = F.shape(key_tensor)
        key_tensor = F.reshape(key_tensor, (-1, key_shape[-1]))
        value_shape = F.shape(value_tensor)
        value_tensor = F.reshape(value_tensor, (-1, value_shape[-1]))
        return query_tensor, key_tensor, value_tensor, F.shape(attention_mask)[0], query_shape

    def _merge_heads(self, x):
        x = self.merger_head_transpose(
            x, (0, 2, 1, 3))
        x_shape = P.Shape()(x)
        new_shape = (-1, x_shape[-2] * x_shape[-1])
        x_merge = self.reshape(x, new_shape)
        return x_merge

    def _attn(self, query, key, value, attention_mask):
        score = self.batch_matmul(query, key)
        score = self.real_div(
            score,
            P.Cast()(self.scale_factor, P.DType()(score)))

        ori_dtype = P.DType()(score)
        score = P.Cast()(score, self.softmax_dtype)

        multiplu_out = self.sub(
            P.Cast()(F.tuple_to_array((1.0,)), P.DType()(score)),
            P.Cast()(attention_mask, P.DType()(score)))
        adder = self.mul(multiplu_out, self.multiply_data)
        attention_scores = self.add(adder, score)
        shape = F.shape(attention_scores)
        attention_probs = self.softmax(
            F.reshape(attention_scores,
                      (shape[0], -1, shape[-1])))
        attention_probs = P.Cast()(attention_probs, ori_dtype)
        attention_probs = F.reshape(attention_probs, shape)

        attention_probs = self.prob_dropout(attention_probs)
        weighted_values = self.batch_matmul(attention_probs, value)
        attention_merge = self._merge_heads(weighted_values)
        return attention_merge


class TransformerEncoderLayer(Cell):
    def __init__(self,
                 batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 hidden_act='gelu'):
        super(TransformerEncoderLayer, self).__init__()
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.layernorm1 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.layernorm2 = _LayerNorm((hidden_size,)).to_float(layernorm_compute_type)
        self.attention = MultiHeadAttention(batch_size=batch_size,
                                            src_seq_length=seq_length,
                                            tgt_seq_length=seq_length,
                                            hidden_size=hidden_size,
                                            num_heads=num_heads,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            attention_dropout_rate=attention_dropout_rate,
                                            softmax_compute_type=softmax_compute_type,
                                            param_init_type=param_init_type)
        self.output = FeedForward(hidden_size=hidden_size,
                                  dropout_rate=hidden_dropout_rate,
                                  ffn_hidden_size=ffn_hidden_size,
                                  param_init_type=param_init_type,
                                  hidden_act=hidden_act)
        self.add = P.Add()
        self.add_3d = P.Add()
        self.dtype = mstype.float16
        self.key_past = None
        self.value_past = None

    def construct(self, x, input_mask, init_reset=True, batch_valid_length=None):
        x_shape = F.shape(x)
        x = F.reshape(x, (-1, x_shape[-1]))
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(input_x, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        x = self.add(x, attention)
        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)
        value_update = None
        key_update = None
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)
        if len(x_shape) == 3:
            output_x = P.Reshape()(output_x, x_shape)
            mlp_logit = P.Reshape()(mlp_logit, x_shape)
            x = P.Reshape()(x, x_shape)
            output = self.add_3d(x, mlp_logit)
        else:
            output = self.add(x, mlp_logit)
            output = F.reshape(output, x_shape)
        return output, layer_present


class TransformerEncoder(Cell):
    def __init__(self,
                 batch_size,
                 num_layers,
                 hidden_size,
                 ffn_hidden_size,
                 seq_length,
                 num_heads,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 hidden_act='gelu',
                 layernorm_compute_type=mstype.float32,
                 softmax_compute_type=mstype.float32,
                 param_init_type=mstype.float32,
                 lambda_func=None,
                 offset=0):
        super(TransformerEncoder, self).__init__()
        self.add = P.Add().shard(((), ()))
        self.aux_loss = Tensor(0.0, mstype.float32)
        self.num_layers = num_layers
        self.blocks = nn.CellList()
        for i in range(num_layers):
            block = TransformerEncoderLayer(hidden_size=hidden_size,
                                            batch_size=batch_size,
                                            ffn_hidden_size=ffn_hidden_size,
                                            seq_length=seq_length,
                                            attention_dropout_rate=attention_dropout_rate,
                                            hidden_dropout_rate=hidden_dropout_rate,
                                            layernorm_compute_type=layernorm_compute_type,
                                            softmax_compute_type=softmax_compute_type,
                                            num_heads=num_heads,
                                            hidden_act=hidden_act,
                                            param_init_type=param_init_type)
            lambda_func(block, layer_id=i, layers=num_layers,
                        offset=offset)
            self.blocks.append(block)

    def construct(self, hidden_states, attention_mask, init_reset=True, batch_valid_length=None):
        present_layer = ()
        for i in range(self.num_layers):
            hidden_states, present = self.blocks[i](hidden_states,
                                                    attention_mask,
                                                    init_reset,
                                                    batch_valid_length)
            present_layer = present_layer + (present,)
        return hidden_states, present_layer


class VocabEmbedding(Cell):
    def __init__(self, vocab_size, embedding_size, param_init='normal'):
        super(VocabEmbedding, self).__init__()
        self.embedding_table = Parameter(initializer(param_init, [vocab_size, embedding_size]),
                                         name='embedding_table', parallel_optimizer=False)
        self.gather = P.GatherV2().shard(((1, 1), (2, 1)))

    def construct(self, input_ids):
        output = self.gather(self.embedding_table, input_ids, 0)
        return output, self.embedding_table.value()


class EmbeddingLayer(nn.Cell):
    def __init__(self):
        super(EmbeddingLayer, self).__init__()
        self.word_embedding = VocabEmbedding(vocab_size=40000, embedding_size=2560)
        self.position_embedding = VocabEmbedding(vocab_size=40000, embedding_size=2560)
        self.add = P.Add()
        self.dropout = nn.Dropout(p=0.1)

    def construct(self, input_ids, input_position, init_reset, batch_valid_length):
        word_embedding, word_table = self.word_embedding(input_ids)
        position_embedding, _ = self.position_embedding(input_position)
        embed = self.add(word_embedding, position_embedding)
        embed = self.dropout(embed)
        return embed, word_table


class QueryLayer(TransformerEncoderLayer):
    def __init__(self, batch_size,
                 hidden_size,
                 ffn_hidden_size,
                 num_heads,
                 seq_length,
                 attention_dropout_rate=0.1,
                 hidden_dropout_rate=0.1,
                 param_init_type=mstype.float32,
                 hidden_act='gelu',
                 softmax_compute_type=mstype.float32):
        super(QueryLayer, self).__init__(batch_size=batch_size,
                                         hidden_size=hidden_size,
                                         ffn_hidden_size=ffn_hidden_size,
                                         num_heads=num_heads,
                                         seq_length=seq_length,
                                         attention_dropout_rate=attention_dropout_rate,
                                         hidden_dropout_rate=hidden_dropout_rate,
                                         param_init_type=param_init_type,
                                         hidden_act=hidden_act,
                                         softmax_compute_type=softmax_compute_type)

    def construct(self, x, query_vector, input_mask, init_reset=True, batch_valid_length=None):
        input_x = self.layernorm1(x)
        input_x = F.cast(input_x, self.dtype)
        attention, layer_present = self.attention(query_vector, input_x, input_x, input_mask,
                                                  self.key_past, self.value_past, batch_valid_length)
        x = self.add(x, attention)
        output_x = self.layernorm2(x)
        output_x = F.cast(output_x, self.dtype)
        mlp_logit = self.output(output_x)

        value_update = None
        key_update = None
        mlp_logit = F.depend(mlp_logit, value_update)
        mlp_logit = F.depend(mlp_logit, key_update)
        output = self.add(x, mlp_logit)
        return output, layer_present


class PanGuHead(Cell):
    def __init__(self,
                 hidden_size,
                 compute_type=mstype.float16):
        super(PanGuHead, self).__init__()
        self.matmul = P.MatMul(transpose_b=True).shard(((2, 1), (1, 1)))
        self.hidden_size = hidden_size
        self.dtype = compute_type
        self.cast = P.Cast()

    def construct(self, state, embed):
        state = P.Reshape()(state, (-1, self.hidden_size))
        logits = self.matmul(self.cast(state, self.dtype), self.cast(embed, self.dtype))
        return logits


class PanguAlphaRawModel(Cell):
    def __init__(self):
        super(PanguAlphaRawModel, self).__init__()
        self.embedding = EmbeddingLayer()
        self.layernorm = _LayerNorm((2560,)).to_float(mstype.float32)
        self.layernorm.set_comm_fusion(4)
        self.embedding.pipeline_stage = 0
        self.blocks = TransformerEncoder(num_layers=1,
                                         batch_size=32,
                                         hidden_size=2560,
                                         ffn_hidden_size=10240,
                                         num_heads=32,
                                         seq_length=1024,
                                         attention_dropout_rate=0.1,
                                         hidden_dropout_rate=0.1,
                                         lambda_func=set_parallel_configure_for_layer,
                                         param_init_type=mstype.float32,
                                         softmax_compute_type=mstype.float16).blocks
        self.top_query_embedding = VocabEmbedding(vocab_size=1024,
                                                  embedding_size=2560)
        self.top_query_embedding.set_comm_fusion(4)
        self.top_query_layer = QueryLayer(batch_size=32,
                                          hidden_size=2560,
                                          ffn_hidden_size=10240,
                                          num_heads=32,
                                          seq_length=1024,
                                          attention_dropout_rate=0.1,
                                          hidden_dropout_rate=0.1,
                                          hidden_act="fast_gelu",
                                          param_init_type=mstype.float32)
        self.top_query_layer.set_comm_fusion(4)
        self.dtype = mstype.float16

    def construct(self, input_ids,
                  input_position,
                  encoder_masks,
                  init_reset=True,
                  batch_valid_length=None):
        embed, word_table = self.embedding(input_ids, input_position, init_reset, batch_valid_length)
        hidden_state = P.Cast()(embed, self.dtype)
        hidden_state = self.reshape_to_2d(hidden_state)
        hidden_state, _ = self.blocks[0](hidden_state, encoder_masks, init_reset, batch_valid_length)
        hidden_state = self.reshape_to_2d(hidden_state)
        encoder_output = self.layernorm(hidden_state)
        encoder_output = P.Cast()(encoder_output, self.dtype)
        top_query_hidden_states, _ = self.top_query_embedding(input_position)
        top_query_hidden_states = self.reshape_to_2d(top_query_hidden_states)
        encoder_output, _ = self.top_query_layer(encoder_output, top_query_hidden_states,
                                                 encoder_masks, init_reset, batch_valid_length)
        return encoder_output, word_table

    def reshape_to_2d(self, x):
        shape = F.shape(x)
        if len(shape) <= 2:
            return x
        x = F.reshape(x, (-1, shape[-1]))
        return x


class PanguAlphaModel(nn.Cell):
    def __init__(self):
        super(PanguAlphaModel, self).__init__()
        self.head = PanGuHead(hidden_size=2560)
        self.backbone = PanguAlphaRawModel()
        self.backbone.embedding.word_embedding.embedding_table.add_pipeline_stage(0)

    def construct(self, input_ids, input_position, attention_mask,
                  init_reset=True, batch_valid_length=None):
        output_states, word_table = self.backbone(input_ids, input_position, attention_mask,
                                                  init_reset, batch_valid_length)
        logits = self.head(output_states, word_table)
        return logits


class CrossEntropyLoss(Cell):
    def __init__(self):
        super(CrossEntropyLoss, self).__init__()
        dp = 2
        mp = 8
        self.sum = P.ReduceSum()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, mstype.float32)
        self.off_value = Tensor(0.0, mstype.float32)
        self.max = P.ArgMaxWithValue(axis=-1, keep_dims=True).shard(
            ((dp, mp),))
        self.eps_const = Tensor(1e-24, mstype.float32)
        self.sub = P.Sub()
        self.exp = P.Exp()
        self.div = P.RealDiv()
        self.log = P.Log()
        self.add = P.Add()
        self.mul = P.Mul()
        self.neg = P.Neg()
        self.sum2 = P.ReduceSum()

        self.mul2 = P.Mul()
        self.add2 = P.Add()
        self.div2 = P.RealDiv()

    def construct(self, logits, label, input_mask):
        logits = F.cast(logits, mstype.float32)
        _, logit_max = self.max(logits)
        logit_sub = self.sub(logits, logit_max)
        logit_exp = self.exp(logit_sub)
        exp_sum = self.sum(logit_exp, -1)
        exp_sum = P.Reshape()(exp_sum, (F.shape(exp_sum)[0], 1))
        softmax_result = self.div(logit_exp, exp_sum)
        log_softmax_result = self.log(self.add(softmax_result, self.eps_const))
        label = P.Reshape()(label, (-1,))
        one_hot_label = self.onehot(label, F.shape(logits)[-1], self.on_value,
                                    self.off_value)
        loss = self.mul(log_softmax_result, one_hot_label)
        loss_unsum = self.neg(loss)
        loss_reduce = self.sum(loss_unsum, -1)
        input_mask = P.Reshape()(input_mask, (-1,))
        numerator = self.sum2(self.mul2(loss_reduce, input_mask))
        denominator = self.add2(
            self.sum2(input_mask),
            P.Cast()(F.tuple_to_array((1e-5,)), mstype.float32))
        loss = self.div2(numerator, denominator)
        return loss


class PanGUAlphaWithLoss(Cell):
    def __init__(self, network, loss):
        super(PanGUAlphaWithLoss, self).__init__(auto_prefix=False)
        self.batch_size = 32
        self.seq_length = 1024
        self.network = network
        self.eod_token = True
        self.loss = loss

        self.slice = P.StridedSlice()
        self.not_equal = P.NotEqual()
        self.len = self.seq_length
        self.slice2 = P.StridedSlice()
        self.micro_batch_step = 1

    def construct(self, input_ids, input_position=None, attention_mask=None):
        tokens = self.slice(input_ids, (0, 0), (self.batch_size, -1), (1, 1))
        input_position = self.slice(input_position, (0, 0), (self.batch_size, self.len), (1, 1))
        decoder_attention_masks = self.slice2(attention_mask, (0, 0, 0), (self.batch_size, self.len, self.len),
                                              (1, 1, 1))
        input_mask = F.cast(self.not_equal(tokens, self.eod_token),
                            mstype.float32)

        logits = self.network(tokens,
                              input_position,
                              decoder_attention_masks)
        labels = self.slice(input_ids, (0, 1), (self.batch_size, self.len + 1),
                            (1, 1))
        labels = P.Reshape()(labels, (-1,))
        input_mask = P.Reshape()(input_mask, (-1,))
        output = self.loss(logits, labels, input_mask)
        return output


def test_pangu_alpha_shard_propagation():
    '''
    Feature: sharding propagatation
    Description: Propagate strategies on pangu_alpha just use a few ops configured stra
    Expectation: Get expected strategies by check several key ops
    '''
    context.reset_auto_parallel_context()
    context.set_auto_parallel_context(parallel_mode="auto_parallel", device_num=16,
                                      search_mode="sharding_propagation")
    set_algo_parameters(elementwise_op_strategy_follow=False, fully_use_devices=False)
    pangu_alpha = PanguAlphaModel()
    loss = CrossEntropyLoss()
    pangu_alpha_loss = PanGUAlphaWithLoss(pangu_alpha, loss)
    net = _VirtualDatasetCell(pangu_alpha_loss)
    input_ids = Tensor(np.ones((2, 1025)), mstype.int32)
    input_position = Tensor(np.ones((2, 1024)), mstype.int32)
    attention_mask = Tensor(np.ones((2, 1024, 1024)), mstype.float16)
    dataset = Dataset(input_ids, input_position, attention_mask)
    model = Model(net)
    model.train(1, dataset, dataset_sink_mode=False)
    stras = _cell_graph_executor._get_shard_strategy(model._train_network)
    for (k, v) in stras.items():
        if re.search("MultiHeadAttention/Add", k) is not None:
            assert v == [[2, 1, 1, 1], [2, 8, 1, 1]]
        elif re.search("ReduceMean", k) is not None:
            assert v == [[2, 1]]
        elif re.search("BatchMatmul", k) is not None:
            assert v == [[2, 8, 1, 1], [2, 8, 1, 1]]
    context.reset_auto_parallel_context()
