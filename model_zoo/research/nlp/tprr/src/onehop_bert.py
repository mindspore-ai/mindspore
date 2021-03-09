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
        self.reducemean_0 = P.ReduceMean(keep_dims=True)
        self.sub_1 = P.Sub()
        self.cast_2 = P.Cast()
        self.cast_2_to = mstype.float32
        self.pow_3 = P.Pow()
        self.pow_3_input_weight = 2.0
        self.reducemean_4 = P.ReduceMean(keep_dims=True)
        self.add_5 = P.Add()
        self.add_5_bias = 9.999999960041972e-13
        self.sqrt_6 = P.Sqrt()
        self.div_7 = P.Div()
        self.mul_8 = P.Mul()
        self.mul_8_w = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.add_9 = P.Add()
        self.add_9_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        opt_reducemean_0 = self.reducemean_0(x, -1)
        opt_sub_1 = self.sub_1(x, opt_reducemean_0)
        opt_cast_2 = self.cast_2(opt_sub_1, self.cast_2_to)
        opt_pow_3 = self.pow_3(opt_cast_2, self.pow_3_input_weight)
        opt_reducemean_4 = self.reducemean_4(opt_pow_3, -1)
        opt_add_5 = self.add_5(opt_reducemean_4, self.add_5_bias)
        opt_sqrt_6 = self.sqrt_6(opt_add_5)
        opt_div_7 = self.div_7(opt_sub_1, opt_sqrt_6)
        opt_mul_8 = self.mul_8(opt_div_7, self.mul_8_w)
        opt_add_9 = self.add_9(opt_mul_8, self.add_9_bias)
        return opt_add_9


class MultiHeadAttn(nn.Cell):
    """multi head attention layer"""
    def __init__(self):
        super(MultiHeadAttn, self).__init__()
        self.matmul_0 = nn.MatMul()
        self.matmul_0.to_float(mstype.float16)
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.matmul_1 = nn.MatMul()
        self.matmul_1.to_float(mstype.float16)
        self.matmul_1_w = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.matmul_2 = nn.MatMul()
        self.matmul_2.to_float(mstype.float16)
        self.matmul_2_w = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.add_3 = P.Add()
        self.add_3_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.add_4 = P.Add()
        self.add_4_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.add_5 = P.Add()
        self.add_5_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)
        self.reshape_6 = P.Reshape()
        self.reshape_6_shape = tuple([BATCH_SIZE, 256, 12, 64])
        self.reshape_7 = P.Reshape()
        self.reshape_7_shape = tuple([BATCH_SIZE, 256, 12, 64])
        self.reshape_8 = P.Reshape()
        self.reshape_8_shape = tuple([BATCH_SIZE, 256, 12, 64])
        self.transpose_9 = P.Transpose()
        self.transpose_10 = P.Transpose()
        self.transpose_11 = P.Transpose()
        self.matmul_12 = nn.MatMul()
        self.matmul_12.to_float(mstype.float16)
        self.div_13 = P.Div()
        self.div_13_w = 8.0
        self.add_14 = P.Add()
        self.softmax_15 = nn.Softmax(axis=3)
        self.matmul_16 = nn.MatMul()
        self.matmul_16.to_float(mstype.float16)
        self.transpose_17 = P.Transpose()
        self.reshape_18 = P.Reshape()
        self.reshape_18_shape = tuple([BATCH_SIZE, 256, 768])
        self.matmul_19 = nn.MatMul()
        self.matmul_19.to_float(mstype.float16)
        self.matmul_19_w = Parameter(Tensor(np.random.uniform(0, 1, (768, 768)).astype(np.float32)), name=None)
        self.add_20 = P.Add()
        self.add_20_bias = Parameter(Tensor(np.random.uniform(0, 1, (768,)).astype(np.float32)), name=None)

    def construct(self, x, x0):
        """construct function"""
        opt_matmul_0 = self.matmul_0(x, self.matmul_0_w)
        opt_matmul_1 = self.matmul_1(x, self.matmul_1_w)
        opt_matmul_2 = self.matmul_2(x, self.matmul_2_w)
        opt_matmul_0 = P.Cast()(opt_matmul_0, mstype.float32)
        opt_matmul_1 = P.Cast()(opt_matmul_1, mstype.float32)
        opt_matmul_2 = P.Cast()(opt_matmul_2, mstype.float32)
        opt_add_3 = self.add_3(opt_matmul_0, self.add_3_bias)
        opt_add_4 = self.add_4(opt_matmul_1, self.add_4_bias)
        opt_add_5 = self.add_5(opt_matmul_2, self.add_5_bias)
        opt_reshape_6 = self.reshape_6(opt_add_3, self.reshape_6_shape)
        opt_reshape_7 = self.reshape_7(opt_add_4, self.reshape_7_shape)
        opt_reshape_8 = self.reshape_8(opt_add_5, self.reshape_8_shape)
        opt_transpose_9 = self.transpose_9(opt_reshape_6, (0, 2, 1, 3))
        opt_transpose_10 = self.transpose_10(opt_reshape_7, (0, 2, 3, 1))
        opt_transpose_11 = self.transpose_11(opt_reshape_8, (0, 2, 1, 3))
        opt_matmul_12 = self.matmul_12(opt_transpose_9, opt_transpose_10)
        opt_matmul_12 = P.Cast()(opt_matmul_12, mstype.float32)
        opt_div_13 = self.div_13(opt_matmul_12, self.div_13_w)
        opt_add_14 = self.add_14(opt_div_13, x0)
        opt_add_14 = P.Cast()(opt_add_14, mstype.float32)
        opt_softmax_15 = self.softmax_15(opt_add_14)
        opt_matmul_16 = self.matmul_16(opt_softmax_15, opt_transpose_11)
        opt_matmul_16 = P.Cast()(opt_matmul_16, mstype.float32)
        opt_transpose_17 = self.transpose_17(opt_matmul_16, (0, 2, 1, 3))
        opt_reshape_18 = self.reshape_18(opt_transpose_17, self.reshape_18_shape)
        opt_matmul_19 = self.matmul_19(opt_reshape_18, self.matmul_19_w)
        opt_matmul_19 = P.Cast()(opt_matmul_19, mstype.float32)
        opt_add_20 = self.add_20(opt_matmul_19, self.add_20_bias)
        return opt_add_20


class Linear(nn.Cell):
    """linear layer"""
    def __init__(self, matmul_0_weight_shape, add_1_bias_shape):
        super(Linear, self).__init__()
        self.matmul_0 = nn.MatMul()
        self.matmul_0.to_float(mstype.float16)
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, matmul_0_weight_shape).astype(np.float32)),
                                    name=None)
        self.add_1 = P.Add()
        self.add_1_bias = Parameter(Tensor(np.random.uniform(0, 1, add_1_bias_shape).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        opt_matmul_0 = self.matmul_0(x, self.matmul_0_w)
        opt_matmul_0 = P.Cast()(opt_matmul_0, mstype.float32)
        opt_add_1 = self.add_1(opt_matmul_0, self.add_1_bias)
        return opt_add_1


class GeLU(nn.Cell):
    """gelu layer"""
    def __init__(self):
        super(GeLU, self).__init__()
        self.div_0 = P.Div()
        self.div_0_w = 1.4142135381698608
        self.erf_1 = P.Erf()
        self.add_2 = P.Add()
        self.add_2_bias = 1.0
        self.mul_3 = P.Mul()
        self.mul_4 = P.Mul()
        self.mul_4_w = 0.5

    def construct(self, x):
        """construct function"""
        opt_div_0 = self.div_0(x, self.div_0_w)
        opt_erf_1 = self.erf_1(opt_div_0)
        opt_add_2 = self.add_2(opt_erf_1, self.add_2_bias)
        opt_mul_3 = self.mul_3(x, opt_add_2)
        opt_mul_4 = self.mul_4(opt_mul_3, self.mul_4_w)
        return opt_mul_4


class TransformerLayer(nn.Cell):
    """transformer layer"""
    def __init__(self, linear3_0_matmul_0_weight_shape, linear3_0_add_1_bias_shape, linear3_1_matmul_0_weight_shape,
                 linear3_1_add_1_bias_shape):
        super(TransformerLayer, self).__init__()
        self.multiheadattn_0 = MultiHeadAttn()
        self.add_0 = P.Add()
        self.layernorm1_0 = LayerNorm()
        self.linear3_0 = Linear(matmul_0_weight_shape=linear3_0_matmul_0_weight_shape,
                                add_1_bias_shape=linear3_0_add_1_bias_shape)
        self.gelu1_0 = GeLU()
        self.linear3_1 = Linear(matmul_0_weight_shape=linear3_1_matmul_0_weight_shape,
                                add_1_bias_shape=linear3_1_add_1_bias_shape)
        self.add_1 = P.Add()
        self.layernorm1_1 = LayerNorm()

    def construct(self, x, x0):
        """construct function"""
        multiheadattn_0_opt = self.multiheadattn_0(x, x0)
        opt_add_0 = self.add_0(multiheadattn_0_opt, x)
        layernorm1_0_opt = self.layernorm1_0(opt_add_0)
        linear3_0_opt = self.linear3_0(layernorm1_0_opt)
        gelu1_0_opt = self.gelu1_0(linear3_0_opt)
        linear3_1_opt = self.linear3_1(gelu1_0_opt)
        opt_add_1 = self.add_1(linear3_1_opt, layernorm1_0_opt)
        layernorm1_1_opt = self.layernorm1_1(opt_add_1)
        return layernorm1_1_opt


class Encoder1_4(nn.Cell):
    """encoder layer"""
    def __init__(self):
        super(Encoder1_4, self).__init__()
        self.module47_0 = TransformerLayer(linear3_0_matmul_0_weight_shape=(768, 3072),
                                           linear3_0_add_1_bias_shape=(3072,),
                                           linear3_1_matmul_0_weight_shape=(3072, 768),
                                           linear3_1_add_1_bias_shape=(768,))
        self.module47_1 = TransformerLayer(linear3_0_matmul_0_weight_shape=(768, 3072),
                                           linear3_0_add_1_bias_shape=(3072,),
                                           linear3_1_matmul_0_weight_shape=(3072, 768),
                                           linear3_1_add_1_bias_shape=(768,))
        self.module47_2 = TransformerLayer(linear3_0_matmul_0_weight_shape=(768, 3072),
                                           linear3_0_add_1_bias_shape=(3072,),
                                           linear3_1_matmul_0_weight_shape=(3072, 768),
                                           linear3_1_add_1_bias_shape=(768,))
        self.module47_3 = TransformerLayer(linear3_0_matmul_0_weight_shape=(768, 3072),
                                           linear3_0_add_1_bias_shape=(3072,),
                                           linear3_1_matmul_0_weight_shape=(3072, 768),
                                           linear3_1_add_1_bias_shape=(768,))

    def construct(self, x, x0):
        """construct function"""
        module47_0_opt = self.module47_0(x, x0)
        module47_1_opt = self.module47_1(module47_0_opt, x0)
        module47_2_opt = self.module47_2(module47_1_opt, x0)
        module47_3_opt = self.module47_3(module47_2_opt, x0)
        return module47_3_opt


class ModelOneHop(nn.Cell):
    """one hop layer"""
    def __init__(self):
        super(ModelOneHop, self).__init__()
        self.expanddims_0 = P.ExpandDims()
        self.expanddims_0_axis = 1
        self.expanddims_3 = P.ExpandDims()
        self.expanddims_3_axis = 2
        self.cast_5 = P.Cast()
        self.cast_5_to = mstype.float32
        self.sub_7 = P.Sub()
        self.sub_7_bias = 1.0
        self.mul_9 = P.Mul()
        self.mul_9_w = -10000.0
        self.gather_1_input_weight = Parameter(Tensor(np.random.uniform(0, 1, (30522, 768)).astype(np.float32)),
                                               name=None)
        self.gather_1_axis = 0
        self.gather_1 = P.Gather()
        self.gather_2_input_weight = Parameter(Tensor(np.random.uniform(0, 1, (2, 768)).astype(np.float32)), name=None)
        self.gather_2_axis = 0
        self.gather_2 = P.Gather()
        self.add_4 = P.Add()
        self.add_4_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 256, 768)).astype(np.float32)), name=None)
        self.add_6 = P.Add()
        self.layernorm1_0 = LayerNorm()
        self.module51_0 = Encoder1_4()
        self.module51_1 = Encoder1_4()
        self.module51_2 = Encoder1_4()
        self.gather_643_input_weight = Tensor(np.array(0))
        self.gather_643_axis = 1
        self.gather_643 = P.Gather()
        self.dense_644 = nn.Dense(in_channels=768, out_channels=768, has_bias=True)
        self.tanh_645 = nn.Tanh()

    def construct(self, input_ids, token_type_ids, attention_mask):
        """construct function"""
        input_ids = self.cast_5(input_ids, mstype.int32)
        token_type_ids = self.cast_5(token_type_ids, mstype.int32)
        attention_mask = self.cast_5(attention_mask, mstype.int32)
        opt_expanddims_0 = self.expanddims_0(attention_mask, self.expanddims_0_axis)
        opt_expanddims_3 = self.expanddims_3(opt_expanddims_0, self.expanddims_3_axis)
        opt_cast_5 = self.cast_5(opt_expanddims_3, self.cast_5_to)
        opt_sub_7 = self.sub_7(self.sub_7_bias, opt_cast_5)
        opt_mul_9 = self.mul_9(opt_sub_7, self.mul_9_w)
        opt_gather_1_axis = self.gather_1_axis
        opt_gather_1 = self.gather_1(self.gather_1_input_weight, input_ids, opt_gather_1_axis)
        opt_gather_2_axis = self.gather_2_axis
        opt_gather_2 = self.gather_2(self.gather_2_input_weight, token_type_ids, opt_gather_2_axis)
        opt_add_4 = self.add_4(opt_gather_1, self.add_4_bias)
        opt_add_6 = self.add_6(opt_add_4, opt_gather_2)
        layernorm1_0_opt = self.layernorm1_0(opt_add_6)
        module51_0_opt = self.module51_0(layernorm1_0_opt, opt_mul_9)
        module51_1_opt = self.module51_1(module51_0_opt, opt_mul_9)
        module51_2_opt = self.module51_2(module51_1_opt, opt_mul_9)
        opt_gather_643_axis = self.gather_643_axis
        opt_gather_643 = self.gather_643(module51_2_opt, self.gather_643_input_weight, opt_gather_643_axis)
        opt_dense_644 = self.dense_644(opt_gather_643)
        opt_tanh_645 = self.tanh_645(opt_dense_644)
        return opt_tanh_645
