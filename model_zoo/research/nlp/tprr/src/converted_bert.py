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
"""
One Hop BERT.
"""

import numpy as np

from mindspore import nn
from mindspore import Tensor, Parameter
import mindspore.common.dtype as mstype
from mindspore.ops import operations as P

BATCH_SIZE = -1


class LayerNorm(nn.Cell):
    """layer norm"""

    def __init__(self):
        super(LayerNorm, self).__init__()
        self.reducemean = P.ReduceMean(keep_dims=True)
        self.sub = P.Sub()
        self.cast = P.Cast()
        self.cast_to = mstype.float32
        self.pow = P.Pow()
        self.pow_weight = 2.0
        self.add = P.Add()
        self.add_bias_0 = 9.999999960041972e-13
        self.sqrt = P.Sqrt()
        self.div = P.Div()
        self.mul = P.Mul()
        self.mul_weight = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.add_bias_1 = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        x_mean = self.reducemean(x, -1)
        x_sub = self.sub(x, x_mean)
        x_sub = self.cast(x_sub, self.cast_to)
        x_pow = self.pow(x_sub, self.pow_weight)
        out_mean = self.reducemean(x_pow, -1)
        out_add = self.add(out_mean, self.add_bias_0)
        out_sqrt = self.sqrt(out_add)
        out_div = self.div(x_sub, out_sqrt)
        out_mul = self.mul(out_div, self.mul_weight)
        output = self.add(out_mul, self.add_bias_1)
        return output


class MultiHeadAttn(nn.Cell):
    """multi head attention layer"""

    def __init__(self, seq_len):
        super(MultiHeadAttn, self).__init__()
        self.matmul = nn.MatMul()
        self.matmul.to_float(mstype.float16)
        self.query = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.key = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.value = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.add = P.Add()
        self.query_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.key_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.value_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.reshape = P.Reshape()
        self.to_shape_0 = tuple([BATCH_SIZE, seq_len, 12, 64])
        self.transpose = P.Transpose()
        self.div = P.Div()
        self.div_w = 8.0
        self.softmax = nn.Softmax(axis=3)
        self.to_shape_1 = tuple([BATCH_SIZE, seq_len, 768])
        self.context_weight = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.context_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)

    def construct(self, input_tensor, attention_mask):
        """construct function"""
        query_output = self.matmul(input_tensor, self.query)
        key_output = self.matmul(input_tensor, self.key)
        value_output = self.matmul(input_tensor, self.value)
        query_output = P.Cast()(query_output, mstype.float32)
        key_output = P.Cast()(key_output, mstype.float32)
        value_output = P.Cast()(value_output, mstype.float32)
        query_output = self.add(query_output, self.query_bias)
        key_output = self.add(key_output, self.key_bias)
        value_output = self.add(value_output, self.value_bias)
        query_layer = self.reshape(query_output, self.to_shape_0)
        key_layer = self.reshape(key_output, self.to_shape_0)
        value_layer = self.reshape(value_output, self.to_shape_0)
        query_layer = self.transpose(query_layer, (0, 2, 1, 3))
        key_layer = self.transpose(key_layer, (0, 2, 3, 1))
        value_layer = self.transpose(value_layer, (0, 2, 1, 3))
        attention_scores = self.matmul(query_layer, key_layer)
        attention_scores = P.Cast()(attention_scores, mstype.float32)
        attention_scores = self.div(attention_scores, self.div_w)
        attention_scores = self.add(attention_scores, attention_mask)
        attention_scores = P.Cast()(attention_scores, mstype.float32)
        attention_probs = self.softmax(attention_scores)
        context_layer = self.matmul(attention_probs, value_layer)
        context_layer = P.Cast()(context_layer, mstype.float32)
        context_layer = self.transpose(context_layer, (0, 2, 1, 3))
        context_layer = self.reshape(context_layer, self.to_shape_1)
        context_layer = self.matmul(context_layer, self.context_weight)
        context_layer = P.Cast()(context_layer, mstype.float32)
        context_layer = self.add(context_layer, self.context_bias)
        return context_layer


class Linear(nn.Cell):
    """linear layer"""

    def __init__(self, w_shape, b_shape):
        super(Linear, self).__init__()
        self.matmul = nn.MatMul()
        self.matmul.to_float(mstype.float16)
        self.w = Parameter(Tensor(np.random.uniform(0, 1, w_shape).astype(np.float32)),
                           name=None)
        self.add = P.Add()
        self.b = Parameter(Tensor(np.random.uniform(0, 1, b_shape).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        output = self.matmul(x, self.w)
        output = P.Cast()(output, mstype.float32)
        output = self.add(output, self.b)
        return output


class GeLU(nn.Cell):
    """gelu layer"""

    def __init__(self):
        super(GeLU, self).__init__()
        self.div = P.Div()
        self.div_w = 1.4142135381698608
        self.erf = P.Erf()
        self.add = P.Add()
        self.add_bias = 1.0
        self.mul = P.Mul()
        self.mul_w = 0.5

    def construct(self, x):
        """construct function"""
        output = self.div(x, self.div_w)
        output = self.erf(output)
        output = self.add(output, self.add_bias)
        output = self.mul(x, output)
        output = self.mul(output, self.mul_w)
        return output


class TransformerLayer(nn.Cell):
    """transformer layer"""

    def __init__(self, seq_len, intermediate_size, intermediate_bias, output_size, output_bias):
        super(TransformerLayer, self).__init__()
        self.attention = MultiHeadAttn(seq_len)
        self.add = P.Add()
        self.layernorm1 = LayerNorm()
        self.intermediate = Linear(w_shape=intermediate_size,
                                   b_shape=intermediate_bias)
        self.gelu = GeLU()
        self.output = Linear(w_shape=output_size,
                             b_shape=output_bias)
        self.layernorm2 = LayerNorm()

    def construct(self, hidden_states, attention_mask):
        """construct function"""
        attention_output = self.attention(hidden_states, attention_mask)
        attention_output = self.add(attention_output, hidden_states)
        attention_output = self.layernorm1(attention_output)
        intermediate_output = self.intermediate(attention_output)
        intermediate_output = self.gelu(intermediate_output)
        output = self.output(intermediate_output)
        output = self.add(output, attention_output)
        output = self.layernorm2(output)
        return output


class BertEncoder(nn.Cell):
    """encoder layer"""

    def __init__(self, seq_len):
        super(BertEncoder, self).__init__()
        self.layer1 = TransformerLayer(seq_len,
                                       intermediate_size=(768, 3072),
                                       intermediate_bias=(3072,),
                                       output_size=(3072, 768),
                                       output_bias=(768,))
        self.layer2 = TransformerLayer(seq_len,
                                       intermediate_size=(768, 3072),
                                       intermediate_bias=(3072,),
                                       output_size=(3072, 768),
                                       output_bias=(768,))
        self.layer3 = TransformerLayer(seq_len,
                                       intermediate_size=(768, 3072),
                                       intermediate_bias=(3072,),
                                       output_size=(3072, 768),
                                       output_bias=(768,))
        self.layer4 = TransformerLayer(seq_len,
                                       intermediate_size=(768, 3072),
                                       intermediate_bias=(3072,),
                                       output_size=(3072, 768),
                                       output_bias=(768,))

    def construct(self, input_tensor, attention_mask):
        """construct function"""
        layer1_output = self.layer1(input_tensor, attention_mask)
        layer2_output = self.layer2(layer1_output, attention_mask)
        layer3_output = self.layer3(layer2_output, attention_mask)
        layer4_output = self.layer4(layer3_output, attention_mask)
        return layer4_output


class ModelOneHop(nn.Cell):
    """one hop layer"""

    def __init__(self, seq_len):
        super(ModelOneHop, self).__init__()
        self.expanddims = P.ExpandDims()
        self.expanddims_axis_0 = 1
        self.expanddims_axis_1 = 2
        self.cast = P.Cast()
        self.cast_to = mstype.float32
        self.sub = P.Sub()
        self.sub_bias = 1.0
        self.mul = P.Mul()
        self.mul_w = -10000.0
        self.input_weight_0 = Parameter(Tensor(np.random.uniform(0, 1, (30522, 768)).astype(np.float32)),
                                        name=None)
        self.gather_axis_0 = 0
        self.gather = P.Gather()
        self.input_weight_1 = Parameter(Tensor(np.random.uniform(0, 1, (2, 768)).astype(np.float32)), name=None)
        self.add = P.Add()
        self.add_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, seq_len, 768)).astype(np.float32)), name=None)
        self.layernorm = LayerNorm()
        self.encoder_layer_1_4 = BertEncoder(seq_len)
        self.encoder_layer_5_8 = BertEncoder(seq_len)
        self.encoder_layer_9_12 = BertEncoder(seq_len)
        self.cls_ids = Tensor(np.array(0))
        self.gather_axis_1 = 1
        self.dense = nn.Dense(in_channels=768, out_channels=768, has_bias=True)
        self.tanh = nn.Tanh()

    def construct(self, input_ids, token_type_ids, attention_mask):
        """construct function"""
        input_ids = self.cast(input_ids, mstype.int32)
        token_type_ids = self.cast(token_type_ids, mstype.int32)
        attention_mask = self.cast(attention_mask, mstype.int32)
        attention_mask = self.expanddims(attention_mask, self.expanddims_axis_0)
        attention_mask = self.expanddims(attention_mask, self.expanddims_axis_1)
        attention_mask = self.cast(attention_mask, self.cast_to)
        attention_mask = self.sub(self.sub_bias, attention_mask)
        attention_mask_matrix = self.mul(attention_mask, self.mul_w)
        word_embeddings = self.gather(self.input_weight_0, input_ids, self.gather_axis_0)
        token_type_embeddings = self.gather(self.input_weight_1, token_type_ids, self.gather_axis_0)
        word_embeddings = self.add(word_embeddings, self.add_bias)
        embedding_output = self.add(word_embeddings, token_type_embeddings)
        embedding_output = self.layernorm(embedding_output)
        encoder_output = self.encoder_layer_1_4(embedding_output, attention_mask_matrix)
        encoder_output = self.encoder_layer_5_8(encoder_output, attention_mask_matrix)
        encoder_output = self.encoder_layer_9_12(encoder_output, attention_mask_matrix)
        cls_output = self.gather(encoder_output, self.cls_ids, self.gather_axis_1)
        pooled_output = self.dense(cls_output)
        pooled_output = self.tanh(pooled_output)

        return pooled_output
