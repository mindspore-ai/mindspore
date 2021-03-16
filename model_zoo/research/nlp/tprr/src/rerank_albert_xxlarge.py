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
    def __init__(self, passthrough_w_0, passthrough_w_1):
        """init function"""
        super(LayerNorm, self).__init__()
        self.reducemean_0 = P.ReduceMean(keep_dims=True)
        self.sub_1 = P.Sub()
        self.pow_2 = P.Pow()
        self.pow_2_input_weight = 2.0
        self.reducemean_3 = P.ReduceMean(keep_dims=True)
        self.add_4 = P.Add()
        self.add_4_bias = 9.999999960041972e-13
        self.sqrt_5 = P.Sqrt()
        self.div_6 = P.Div()
        self.mul_7 = P.Mul()
        self.mul_7_w = passthrough_w_0
        self.add_8 = P.Add()
        self.add_8_bias = passthrough_w_1

    def construct(self, x):
        """construct function"""
        opt_reducemean_0 = self.reducemean_0(x, -1)
        opt_sub_1 = self.sub_1(x, opt_reducemean_0)
        opt_pow_2 = self.pow_2(opt_sub_1, self.pow_2_input_weight)
        opt_reducemean_3 = self.reducemean_3(opt_pow_2, -1)
        opt_add_4 = self.add_4(opt_reducemean_3, self.add_4_bias)
        opt_sqrt_5 = self.sqrt_5(opt_add_4)
        opt_div_6 = self.div_6(opt_sub_1, opt_sqrt_5)
        opt_mul_7 = self.mul_7(opt_div_6, self.mul_7_w)
        opt_add_8 = self.add_8(opt_mul_7, self.add_8_bias)
        return opt_add_8


class Linear(nn.Cell):
    """Linear layer"""
    def __init__(self, matmul_0_w_shape, passthrough_w_0):
        """init function"""
        super(Linear, self).__init__()
        self.matmul_0 = nn.MatMul()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, matmul_0_w_shape).astype(np.float32)), name=None)
        self.add_1 = P.Add()
        self.add_1_bias = passthrough_w_0

    def construct(self, x):
        """construct function"""
        opt_matmul_0 = self.matmul_0(ops.Cast()(x, dst_type), ops.Cast()(self.matmul_0_w, dst_type))
        opt_add_1 = self.add_1(ops.Cast()(opt_matmul_0, dst_type2), self.add_1_bias)
        return opt_add_1


class MultiHeadAttn(nn.Cell):
    """Multi-head attention layer"""
    def __init__(self, batch_size, passthrough_w_0, passthrough_w_1, passthrough_w_2):
        """init function"""
        super(MultiHeadAttn, self).__init__()
        self.batch_size = batch_size
        self.matmul_0 = nn.MatMul()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)), name=None)
        self.matmul_1 = nn.MatMul()
        self.matmul_1_w = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)), name=None)
        self.matmul_2 = nn.MatMul()
        self.matmul_2_w = Parameter(Tensor(np.random.uniform(0, 1, (4096, 4096)).astype(np.float32)), name=None)
        self.add_3 = P.Add()
        self.add_3_bias = passthrough_w_0
        self.add_4 = P.Add()
        self.add_4_bias = passthrough_w_1
        self.add_5 = P.Add()
        self.add_5_bias = passthrough_w_2
        self.reshape_6 = P.Reshape()
        self.reshape_6_shape = tuple([batch_size, 512, 64, 64])
        self.reshape_7 = P.Reshape()
        self.reshape_7_shape = tuple([batch_size, 512, 64, 64])
        self.reshape_8 = P.Reshape()
        self.reshape_8_shape = tuple([batch_size, 512, 64, 64])
        self.transpose_9 = P.Transpose()
        self.transpose_10 = P.Transpose()
        self.transpose_11 = P.Transpose()
        self.matmul_12 = nn.MatMul()
        self.div_13 = P.Div()
        self.div_13_w = 8.0
        self.add_14 = P.Add()
        self.softmax_15 = nn.Softmax(axis=3)
        self.matmul_16 = nn.MatMul()
        self.transpose_17 = P.Transpose()
        self.matmul_18 = P.MatMul()
        self.matmul_18_weight = Parameter(Tensor(np.random.uniform(0, 1, (64, 64, 4096)).astype(np.float32)), name=None)
        self.add_19 = P.Add()
        self.add_19_bias = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)

    def construct(self, x, x0):
        """construct function"""
        opt_matmul_0 = self.matmul_0(ops.Cast()(x, dst_type), ops.Cast()(self.matmul_0_w, dst_type))
        opt_matmul_1 = self.matmul_1(ops.Cast()(x, dst_type), ops.Cast()(self.matmul_1_w, dst_type))
        opt_matmul_2 = self.matmul_2(ops.Cast()(x, dst_type), ops.Cast()(self.matmul_2_w, dst_type))
        opt_add_3 = self.add_3(ops.Cast()(opt_matmul_0, dst_type2), self.add_3_bias)
        opt_add_4 = self.add_4(ops.Cast()(opt_matmul_1, dst_type2), self.add_4_bias)
        opt_add_5 = self.add_5(ops.Cast()(opt_matmul_2, dst_type2), self.add_5_bias)
        opt_reshape_6 = self.reshape_6(opt_add_3, self.reshape_6_shape)
        opt_reshape_7 = self.reshape_7(opt_add_4, self.reshape_7_shape)
        opt_reshape_8 = self.reshape_8(opt_add_5, self.reshape_8_shape)
        opt_transpose_9 = self.transpose_9(opt_reshape_6, (0, 2, 1, 3))
        opt_transpose_10 = self.transpose_10(opt_reshape_7, (0, 2, 3, 1))
        opt_transpose_11 = self.transpose_11(opt_reshape_8, (0, 2, 1, 3))
        opt_matmul_12 = self.matmul_12(ops.Cast()(opt_transpose_9, dst_type), ops.Cast()(opt_transpose_10, dst_type))
        opt_div_13 = self.div_13(ops.Cast()(opt_matmul_12, dst_type2), ops.Cast()(self.div_13_w, dst_type2))
        opt_add_14 = self.add_14(opt_div_13, x0)
        opt_softmax_15 = self.softmax_15(opt_add_14)
        opt_matmul_16 = self.matmul_16(ops.Cast()(opt_softmax_15, dst_type), ops.Cast()(opt_transpose_11, dst_type))
        opt_transpose_17 = self.transpose_17(ops.Cast()(opt_matmul_16, dst_type2), (0, 2, 1, 3))
        opt_matmul_18 = self.matmul_18(ops.Cast()(opt_transpose_17, dst_type).view(self.batch_size * 512, -1),
                                       ops.Cast()(self.matmul_18_weight, dst_type).view(-1, 4096))\
            .view(self.batch_size, 512, 4096)
        opt_add_19 = self.add_19(ops.Cast()(opt_matmul_18, dst_type2), self.add_19_bias)
        return opt_add_19


class NewGeLU(nn.Cell):
    """Gelu layer"""
    def __init__(self):
        """init function"""
        super(NewGeLU, self).__init__()
        self.mul_0 = P.Mul()
        self.mul_0_w = 0.5
        self.pow_1 = P.Pow()
        self.pow_1_input_weight = 3.0
        self.mul_2 = P.Mul()
        self.mul_2_w = 0.044714998453855515
        self.add_3 = P.Add()
        self.mul_4 = P.Mul()
        self.mul_4_w = 0.7978845834732056
        self.tanh_5 = nn.Tanh()
        self.add_6 = P.Add()
        self.add_6_bias = 1.0
        self.mul_7 = P.Mul()

    def construct(self, x):
        """construct function"""
        opt_mul_0 = self.mul_0(x, self.mul_0_w)
        opt_pow_1 = self.pow_1(x, self.pow_1_input_weight)
        opt_mul_2 = self.mul_2(opt_pow_1, self.mul_2_w)
        opt_add_3 = self.add_3(x, opt_mul_2)
        opt_mul_4 = self.mul_4(opt_add_3, self.mul_4_w)
        opt_tanh_5 = self.tanh_5(opt_mul_4)
        opt_add_6 = self.add_6(opt_tanh_5, self.add_6_bias)
        opt_mul_7 = self.mul_7(opt_mul_0, opt_add_6)
        return opt_mul_7


class TransformerLayerWithLayerNorm(nn.Cell):
    """Transformer layer with LayerNOrm"""
    def __init__(self, batch_size, linear3_0_matmul_0_w_shape, linear3_1_matmul_0_w_shape, passthrough_w_0,
                 passthrough_w_1, passthrough_w_2, passthrough_w_3, passthrough_w_4, passthrough_w_5, passthrough_w_6):
        """init function"""
        super(TransformerLayerWithLayerNorm, self).__init__()
        self.multiheadattn_0 = MultiHeadAttn(batch_size=batch_size,
                                             passthrough_w_0=passthrough_w_0,
                                             passthrough_w_1=passthrough_w_1,
                                             passthrough_w_2=passthrough_w_2)
        self.add_0 = P.Add()
        self.layernorm1_0 = LayerNorm(passthrough_w_0=passthrough_w_3, passthrough_w_1=passthrough_w_4)
        self.linear3_0 = Linear(matmul_0_w_shape=linear3_0_matmul_0_w_shape, passthrough_w_0=passthrough_w_5)
        self.newgelu2_0 = NewGeLU()
        self.linear3_1 = Linear(matmul_0_w_shape=linear3_1_matmul_0_w_shape, passthrough_w_0=passthrough_w_6)
        self.add_1 = P.Add()

    def construct(self, x, x0):
        """construct function"""
        multiheadattn_0_opt = self.multiheadattn_0(x, x0)
        opt_add_0 = self.add_0(x, multiheadattn_0_opt)
        layernorm1_0_opt = self.layernorm1_0(opt_add_0)
        linear3_0_opt = self.linear3_0(layernorm1_0_opt)
        newgelu2_0_opt = self.newgelu2_0(linear3_0_opt)
        linear3_1_opt = self.linear3_1(newgelu2_0_opt)
        opt_add_1 = self.add_1(linear3_1_opt, layernorm1_0_opt)
        return opt_add_1


class Rerank_Albert(nn.Cell):
    """Albert model for rerank"""
    def __init__(self, batch_size):
        """init function"""
        super(Rerank_Albert, self).__init__()
        self.passthrough_w_0 = Parameter(Tensor(np.random.uniform(0, 1, (128,)).astype(np.float32)), name=None)
        self.passthrough_w_1 = Parameter(Tensor(np.random.uniform(0, 1, (128,)).astype(np.float32)), name=None)
        self.passthrough_w_2 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_3 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_4 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_5 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_6 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_7 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_8 = Parameter(Tensor(np.random.uniform(0, 1, (16384,)).astype(np.float32)), name=None)
        self.passthrough_w_9 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_10 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
        self.passthrough_w_11 = Parameter(Tensor(np.random.uniform(0, 1, (4096,)).astype(np.float32)), name=None)
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
        self.gather_1_input_weight = Parameter(Tensor(np.random.uniform(0, 1, (30005, 128)).astype(np.float32)),
                                               name=None)
        self.gather_1_axis = 0
        self.gather_1 = P.Gather()
        self.gather_2_input_weight = Parameter(Tensor(np.random.uniform(0, 1, (2, 128)).astype(np.float32)), name=None)
        self.gather_2_axis = 0
        self.gather_2 = P.Gather()
        self.add_4 = P.Add()
        self.add_4_bias = Parameter(Tensor(np.random.uniform(0, 1, (1, 512, 128)).astype(np.float32)), name=None)
        self.add_6 = P.Add()
        self.layernorm1_0 = LayerNorm(passthrough_w_0=self.passthrough_w_0, passthrough_w_1=self.passthrough_w_1)
        self.linear3_0 = Linear(matmul_0_w_shape=(128, 4096), passthrough_w_0=self.passthrough_w_2)
        self.module34_0 = TransformerLayerWithLayerNorm(batch_size=batch_size,
                                                        linear3_0_matmul_0_w_shape=(4096, 16384),
                                                        linear3_1_matmul_0_w_shape=(16384, 4096),
                                                        passthrough_w_0=self.passthrough_w_3,
                                                        passthrough_w_1=self.passthrough_w_4,
                                                        passthrough_w_2=self.passthrough_w_5,
                                                        passthrough_w_3=self.passthrough_w_6,
                                                        passthrough_w_4=self.passthrough_w_7,
                                                        passthrough_w_5=self.passthrough_w_8,
                                                        passthrough_w_6=self.passthrough_w_9)
        self.layernorm1_1 = LayerNorm(passthrough_w_0=self.passthrough_w_10, passthrough_w_1=self.passthrough_w_11)

    def construct(self, input_ids, attention_mask, token_type_ids):
        """construct function"""
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
        linear3_0_opt = self.linear3_0(layernorm1_0_opt)
        opt = self.module34_0(linear3_0_opt, opt_mul_9)
        opt = self.layernorm1_1(opt)
        for _ in range(11):
            opt = self.module34_0(opt, opt_mul_9)
            opt = self.layernorm1_1(opt)
        return opt
