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
"""downstream Model for reader"""

import numpy as np
from mindspore import nn, ops
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
from mindspore import dtype as mstype


dst_type = mstype.float16
dst_type2 = mstype.float32


class Linear(nn.Cell):
    """module of reader downstream"""
    def __init__(self, linear_weight_shape, linear_bias_shape):
        """init function"""
        super(Linear, self).__init__()
        self.matmul = nn.MatMul()
        self.matmul_w = Parameter(Tensor(np.random.uniform(0, 1, linear_weight_shape).astype(np.float32)),
                                  name=None)
        self.add = P.Add()
        self.add_bias = Parameter(Tensor(np.random.uniform(0, 1, linear_bias_shape).astype(np.float32)), name=None)
        self.relu = nn.ReLU()

    def construct(self, hidden_state):
        """construct function"""
        output = self.matmul(ops.Cast()(hidden_state, dst_type), ops.Cast()(self.matmul_w, dst_type))
        output = self.add(ops.Cast()(output, dst_type2), self.add_bias)
        output = self.relu(output)
        return output


class BertLayerNorm(nn.Cell):
    """Normalization module of reader downstream"""
    def __init__(self, bert_layer_norm_weight_shape, bert_layer_norm_bias_shape, eps=1e-12):
        """init function"""
        super(BertLayerNorm, self).__init__()
        self.reducemean = P.ReduceMean(keep_dims=True)
        self.sub = P.Sub()
        self.pow = P.Pow()
        self.add = P.Add()
        self.sqrt = P.Sqrt()
        self.div = P.Div()
        self.mul = P.Mul()
        self.variance_epsilon = eps
        self.bert_layer_norm_weight = Parameter(Tensor(np.random.uniform(0, 1, bert_layer_norm_weight_shape)
                                                       .astype(np.float32)), name=None)
        self.bert_layer_norm_bias = Parameter(Tensor(np.random.uniform(0, 1, bert_layer_norm_bias_shape)
                                                     .astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        u = self.reducemean(x, -1)
        s = self.reducemean(self.pow(self.sub(x, u), 2), -1)
        x = self.div(self.sub(x, u), self.sqrt(self.add(s, self.variance_epsilon)))
        output = self.mul(self.bert_layer_norm_weight, x)
        output = self.add(output, self.bert_layer_norm_bias)
        return output


class SupportingOutputLayer(nn.Cell):
    """module of reader downstream"""
    def __init__(self, linear_1_weight_shape, linear_1_bias_shape, bert_layer_norm_weight_shape,
                 bert_layer_norm_bias_shape):
        """init function"""
        super(SupportingOutputLayer, self).__init__()
        self.linear_1 = Linear(linear_weight_shape=linear_1_weight_shape,
                               linear_bias_shape=linear_1_bias_shape)
        self.bert_layer_norm = BertLayerNorm(bert_layer_norm_weight_shape=bert_layer_norm_weight_shape,
                                             bert_layer_norm_bias_shape=bert_layer_norm_bias_shape)
        self.matmul = nn.MatMul()
        self.matmul_w = Parameter(Tensor(np.random.uniform(0, 1, (8192, 1)).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        output = self.linear_1(x)
        output = self.bert_layer_norm(output)
        output = self.matmul(ops.Cast()(output, dst_type), ops.Cast()(self.matmul_w, dst_type))
        return ops.Cast()(output, dst_type2)


class PosOutputLayer(nn.Cell):
    """module of reader downstream"""
    def __init__(self, linear_weight_shape, linear_bias_shape, bert_layer_norm_weight_shape,
                 bert_layer_norm_bias_shape):
        """init function"""
        super(PosOutputLayer, self).__init__()
        self.linear_1 = Linear(linear_weight_shape=linear_weight_shape,
                               linear_bias_shape=linear_bias_shape)
        self.bert_layer_norm = BertLayerNorm(bert_layer_norm_weight_shape=bert_layer_norm_weight_shape,
                                             bert_layer_norm_bias_shape=bert_layer_norm_bias_shape)
        self.matmul = nn.MatMul()
        self.linear_2_weight = Parameter(Tensor(np.random.uniform(0, 1, (4096, 1)).astype(np.float32)), name=None)
        self.add = P.Add()
        self.linear_2_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)

    def construct(self, state):
        """construct function"""
        output = self.linear_1(state)
        output = self.bert_layer_norm(output)
        output = self.matmul(ops.Cast()(output, dst_type), ops.Cast()(self.linear_2_weight, dst_type))
        output = self.add(ops.Cast()(output, dst_type2), self.linear_2_bias)
        return output


class MaskInvalidPos(nn.Cell):
    """module of reader downstream"""
    def __init__(self):
        """init function"""
        super(MaskInvalidPos, self).__init__()
        self.squeeze = P.Squeeze(2)
        self.sub = P.Sub()
        self.mul = P.Mul()

    def construct(self, pos_pred, context_mask):
        """construct function"""
        output = self.squeeze(pos_pred)
        invalid_pos_mask = self.mul(self.sub(1.0, context_mask), 1e30)
        output = self.sub(output, invalid_pos_mask)
        return output


class Reader_Downstream(nn.Cell):
    """Downstream model for reader"""
    def __init__(self):
        """init function"""
        super(Reader_Downstream, self).__init__()

        self.add = P.Add()
        self.para_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)
        self.para_output_layer = SupportingOutputLayer(linear_1_weight_shape=(4096, 8192),
                                                       linear_1_bias_shape=(8192,),
                                                       bert_layer_norm_weight_shape=(8192,),
                                                       bert_layer_norm_bias_shape=(8192,))
        self.sent_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)
        self.sent_output_layer = SupportingOutputLayer(linear_1_weight_shape=(4096, 8192),
                                                       linear_1_bias_shape=(8192,),
                                                       bert_layer_norm_weight_shape=(8192,),
                                                       bert_layer_norm_bias_shape=(8192,))

        self.start_output_layer = PosOutputLayer(linear_weight_shape=(4096, 4096),
                                                 linear_bias_shape=(4096,),
                                                 bert_layer_norm_weight_shape=(4096,),
                                                 bert_layer_norm_bias_shape=(4096,))
        self.end_output_layer = PosOutputLayer(linear_weight_shape=(4096, 4096),
                                               linear_bias_shape=(4096,),
                                               bert_layer_norm_weight_shape=(4096,),
                                               bert_layer_norm_bias_shape=(4096,))
        self.mask_invalid_pos = MaskInvalidPos()
        self.gather_input_weight = Tensor(np.array(0))
        self.gather = P.Gather()
        self.type_linear_1 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu = nn.ReLU()

        self.bert_layer_norm = BertLayerNorm(bert_layer_norm_weight_shape=(4096,), bert_layer_norm_bias_shape=(4096,))
        self.type_linear_2 = nn.Dense(in_channels=4096, out_channels=3, has_bias=True)

    def construct(self, para_state, sent_state, state, context_mask):
        """construct function"""
        para_logit = self.para_output_layer(para_state)
        para_logit = self.add(para_logit, self.para_bias)
        sent_logit = self.sent_output_layer(sent_state)
        sent_logit = self.add(sent_logit, self.sent_bias)

        start = self.start_output_layer(state)
        start = self.mask_invalid_pos(start, context_mask)

        end = self.end_output_layer(state)
        end = self.mask_invalid_pos(end, context_mask)

        cls_emb = self.gather(state, self.gather_input_weight, 1)
        q_type = self.type_linear_1(cls_emb)
        q_type = self.relu(q_type)
        q_type = self.bert_layer_norm(q_type)
        q_type = self.type_linear_2(q_type)
        return q_type, start, end, para_logit, sent_logit
