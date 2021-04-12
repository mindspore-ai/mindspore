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
"""albert-xxlarge Model for reranker"""

import numpy as np
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore import dtype as mstype


dst_type = mstype.float16
dst_type2 = mstype.float32


class LayerNorm(nn.Cell):
    """LayerNorm layer"""
    def __init__(self, layer_norm_weight, layer_norm_bias):
        """init function"""
        super(LayerNorm, self).__init__()
        self.reducemean = P.ReduceMean(keep_dims=True)
        self.sub = P.Sub()
        self.pow = P.Pow()
        self.add = P.Add()
        self.sqrt = P.Sqrt()
        self.div = P.Div()
        self.mul = P.Mul()
        self.layer_norm_weight = layer_norm_weight
        self.layer_norm_bias = layer_norm_bias

    def construct(self, x):
        """construct function"""
        diff_ex = self.sub(x, self.reducemean(x, -1))
        var_x = self.reducemean(self.pow(diff_ex, 2.0), -1)
        output = self.div(diff_ex, self.sqrt(self.add(var_x, 1e-12)))
        output = self.add(self.mul(output, self.layer_norm_weight), self.layer_norm_bias)
        return output


class Linear(nn.Cell):
    """Linear layer"""
    def __init__(self, linear_weight_shape, linear_bias):
        """init function"""
        super(Linear, self).__init__()
        self.matmul = nn.MatMul()
        self.add = P.Add()
        self.weight = Parameter(Tensor(np.random.uniform(0, 1, linear_weight_shape).astype(np.float32)), name=None)
        self.bias = linear_bias

    def construct(self, input_x):
        """construct function"""
        output = self.matmul(ops.Cast()(input_x, dst_type), ops.Cast()(self.weight, dst_type))
        output = self.add(ops.Cast()(output, dst_type2), self.bias)
        return output


class MultiHeadAttn(nn.Cell):
    """Multi-head attention layer"""
    def __init__(self, batch_size, query_linear_bias, key_linear_bias, value_linear_bias):
        """init function"""
        super(MultiHeadAttn, self).__init__()
        self.batch_size = batch_size
        self.matmul = nn.MatMul()
        self.add = P.Add()
        self.reshape = P.Reshape()
        self.transpose = P.Transpose()
        self.div = P.Div()
        self.softmax = nn.Softmax(axis=3)

        self.query_linear_weight = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)),
                                             name=None)
        self.query_linear_bias = query_linear_bias

        self.key_linear_weight = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)),
                                           name=None)
        self.key_linear_bias = key_linear_bias

        self.value_linear_weight = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)),
                                             name=None)
        self.value_linear_bias = value_linear_bias

        self.reshape_shape = tuple([batch_size, 512, 64, 64])

        self.w = Parameter(Tensor(np.random.uniform(0, 1, (64, 64, 4096)).astype(np.float32)), name=None)
        self.b = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)

    def construct(self, hidden_states, extended_attention_mask):
        """construct function"""
        mixed_query_layer = self.matmul(ops.Cast()(hidden_states, dst_type),
                                        ops.Cast()(self.query_linear_weight, dst_type))
        mixed_query_layer = self.add(ops.Cast()(mixed_query_layer, dst_type2), self.query_linear_bias)

        mixed_key_layer = self.matmul(ops.Cast()(hidden_states, dst_type),
                                      ops.Cast()(self.key_linear_weight, dst_type))
        mixed_key_layer = self.add(ops.Cast()(mixed_key_layer, dst_type2), self.key_linear_bias)

        mixed_value_layer = self.matmul(ops.Cast()(hidden_states, dst_type),
                                        ops.Cast()(self.value_linear_weight, dst_type))
        mixed_value_layer = self.add(ops.Cast()(mixed_value_layer, dst_type2), self.value_linear_bias)

        query_layer = self.reshape(mixed_query_layer, self.reshape_shape)
        key_layer = self.reshape(mixed_key_layer, self.reshape_shape)
        value_layer = self.reshape(mixed_value_layer, self.reshape_shape)

        query_layer = self.transpose(query_layer, (0, 2, 1, 3))
        key_layer = self.transpose(key_layer, (0, 2, 3, 1))
        value_layer = self.transpose(value_layer, (0, 2, 1, 3))

        attention_scores = self.matmul(ops.Cast()(query_layer, dst_type), ops.Cast()(key_layer, dst_type))
        attention_scores = self.div(ops.Cast()(attention_scores, dst_type2), ops.Cast()(8.0, dst_type2))
        attention_scores = self.add(attention_scores, extended_attention_mask)

        attention_probs = self.softmax(attention_scores)
        context_layer = self.matmul(ops.Cast()(attention_probs, dst_type), ops.Cast()(value_layer, dst_type))
        context_layer = self.transpose(ops.Cast()(context_layer, dst_type2), (0, 2, 1, 3))

        projected_context_layer = self.matmul(ops.Cast()(context_layer, dst_type).view(self.batch_size * 512, -1),
                                              ops.Cast()(self.w, dst_type).view(-1, 4096))\
            .view(self.batch_size, 512, 4096)
        projected_context_layer = self.add(ops.Cast()(projected_context_layer, dst_type2), self.b)
        return projected_context_layer


class NewGeLU(nn.Cell):
    """Gelu layer"""
    def __init__(self):
        """init function"""
        super(NewGeLU, self).__init__()
        self.mul = P.Mul()
        self.pow = P.Pow()
        self.mul = P.Mul()
        self.add = P.Add()
        self.tanh = nn.Tanh()

    def construct(self, x):
        """construct function"""
        output = self.mul(self.add(x, self.mul(self.pow(x, 3.0), 0.044714998453855515)), 0.7978845834732056)
        output = self.tanh(output)
        output = self.mul(self.mul(x, 0.5), self.add(output, 1.0))
        return output


class AlbertTransformer(nn.Cell):
    """Transformer layer with LayerNOrm"""
    def __init__(self, batch_size, ffn_weight_shape, ffn_output_weight_shape, query_linear_bias,
                 key_linear_bias, value_linear_bias, layernorm_weight, layernorm_bias, ffn_bias, ffn_output_bias):
        """init function"""
        super(AlbertTransformer, self).__init__()
        self.multiheadattn = MultiHeadAttn(batch_size=batch_size,
                                           query_linear_bias=query_linear_bias,
                                           key_linear_bias=key_linear_bias,
                                           value_linear_bias=value_linear_bias)
        self.add = P.Add()
        self.layernorm = LayerNorm(layer_norm_weight=layernorm_weight, layer_norm_bias=layernorm_bias)
        self.ffn = Linear(linear_weight_shape=ffn_weight_shape, linear_bias=ffn_bias)
        self.newgelu = NewGeLU()
        self.ffn_output = Linear(linear_weight_shape=ffn_output_weight_shape, linear_bias=ffn_output_bias)
        self.add_1 = P.Add()

    def construct(self, hidden_states, extended_attention_mask):
        """construct function"""
        attention_output = self.multiheadattn(hidden_states, extended_attention_mask)
        hidden_states = self.add(hidden_states, attention_output)
        hidden_states = self.layernorm(hidden_states)
        ffn_output = self.ffn(hidden_states)
        ffn_output = self.newgelu(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        hidden_states = self.add_1(ffn_output, hidden_states)
        return hidden_states


class Albert(nn.Cell):
    """Albert model for rerank"""
    def __init__(self, batch_size):
        """init function"""
        super(Albert, self).__init__()
        self.expanddims = P.ExpandDims()
        self.cast = P.Cast()
        self.sub = P.Sub()
        self.mul = P.Mul()
        self.gather = P.Gather()
        self.add = P.Add()

        self.layernorm_1_weight = Parameter(Tensor(np.random.uniform(0, 1, (128,)).astype(np.float32)), name=None)
        self.layernorm_1_bias = Parameter(Tensor(np.random.uniform(0, 1, (128,)).astype(np.float32)), name=None)
        self.embedding_hidden_mapping_in_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)),
                                                          name=None)
        self.query_linear_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.key_linear_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.value_linear_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.albert_transformer_layernorm_w = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)),
                                                        name=None)
        self.albert_transformer_layernorm_b = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)),
                                                        name=None)
        self.ffn_bias = Parameter(Tensor(np.random.uniform(0, 1, (16384,)).astype(np.float32)), name=None)
        self.ffn_output_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.layernorm_2_weight = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.layernorm_2_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)

        self.word_embeddings = Parameter(Tensor(np.random.uniform(0, 1, (30005, 128)).astype(np.float32)), name=None)
        self.token_type_embeddings = Parameter(Tensor(np.random.uniform(0, 1, (2, 128)).astype(np.float32)), name=None)

        self.position_embeddings = Parameter(Tensor(np.random.uniform(0, 1, (1, 512, 128)).astype(np.float32)),
                                             name=None)

        self.layernorm_1 = LayerNorm(layer_norm_weight=self.layernorm_1_weight, layer_norm_bias=self.layernorm_1_bias)
        self.embedding_hidden_mapping_in = Linear(linear_weight_shape=(128, 4096),
                                                  linear_bias=self.embedding_hidden_mapping_in_bias)

        self.albert_transformer = AlbertTransformer(batch_size=batch_size,
                                                    ffn_weight_shape=(4096, 16384),
                                                    ffn_output_weight_shape=(16384, 4096),
                                                    query_linear_bias=self.query_linear_bias,
                                                    key_linear_bias=self.key_linear_bias,
                                                    value_linear_bias=self.value_linear_bias,
                                                    layernorm_weight=self.albert_transformer_layernorm_w,
                                                    layernorm_bias=self.albert_transformer_layernorm_b,
                                                    ffn_bias=self.ffn_bias,
                                                    ffn_output_bias=self.ffn_output_bias)
        self.layernorm_2 = LayerNorm(layer_norm_weight=self.layernorm_2_weight, layer_norm_bias=self.layernorm_2_bias)

    def construct(self, input_ids, attention_mask, token_type_ids):
        """construct function"""
        extended_attention_mask = self.expanddims(attention_mask, 1)
        extended_attention_mask = self.expanddims(extended_attention_mask, 2)
        extended_attention_mask = self.cast(extended_attention_mask, mstype.float32)
        extended_attention_mask = self.mul(self.sub(1.0, extended_attention_mask), -10000.0)

        inputs_embeds = self.gather(self.word_embeddings, input_ids, 0)
        token_type_embeddings = self.gather(self.token_type_embeddings, token_type_ids, 0)
        embeddings = self.add(self.add(inputs_embeds, self.position_embeddings), token_type_embeddings)
        embeddings = self.layernorm_1(embeddings)

        hidden_states = self.embedding_hidden_mapping_in(embeddings)

        for _ in range(12):
            hidden_states = self.albert_transformer(hidden_states, extended_attention_mask)
            hidden_states = self.layernorm_2(hidden_states)
        return hidden_states
