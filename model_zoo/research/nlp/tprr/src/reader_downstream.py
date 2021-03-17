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


class Module15(nn.Cell):
    """module of reader downstream"""
    def __init__(self, matmul_0_weight_shape, add_1_bias_shape):
        """init function"""
        super(Module15, self).__init__()
        self.matmul_0 = nn.MatMul()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, matmul_0_weight_shape).astype(np.float32)),
                                    name=None)
        self.add_1 = P.Add()
        self.add_1_bias = Parameter(Tensor(np.random.uniform(0, 1, add_1_bias_shape).astype(np.float32)), name=None)
        self.relu_2 = nn.ReLU()

    def construct(self, x):
        """construct function"""
        opt_matmul_0 = self.matmul_0(ops.Cast()(x, dst_type), ops.Cast()(self.matmul_0_w, dst_type))
        opt_add_1 = self.add_1(ops.Cast()(opt_matmul_0, dst_type2), self.add_1_bias)
        opt_relu_2 = self.relu_2(opt_add_1)
        return opt_relu_2


class NormModule(nn.Cell):
    """Normalization module of reader downstream"""
    def __init__(self, mul_8_w_shape, add_9_bias_shape):
        """init function"""
        super(NormModule, self).__init__()
        self.reducemean_0 = P.ReduceMean(keep_dims=True)
        self.sub_1 = P.Sub()
        self.sub_2 = P.Sub()
        self.pow_3 = P.Pow()
        self.pow_3_input_weight = 2.0
        self.reducemean_4 = P.ReduceMean(keep_dims=True)
        self.add_5 = P.Add()
        self.add_5_bias = 9.999999960041972e-13
        self.sqrt_6 = P.Sqrt()
        self.div_7 = P.Div()
        self.mul_8 = P.Mul()
        self.mul_8_w = Parameter(Tensor(np.random.uniform(0, 1, mul_8_w_shape).astype(np.float32)), name=None)
        self.add_9 = P.Add()
        self.add_9_bias = Parameter(Tensor(np.random.uniform(0, 1, add_9_bias_shape).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        opt_reducemean_0 = self.reducemean_0(x, -1)
        opt_sub_1 = self.sub_1(x, opt_reducemean_0)
        opt_sub_2 = self.sub_2(x, opt_reducemean_0)
        opt_pow_3 = self.pow_3(opt_sub_1, self.pow_3_input_weight)
        opt_reducemean_4 = self.reducemean_4(opt_pow_3, -1)
        opt_add_5 = self.add_5(opt_reducemean_4, self.add_5_bias)
        opt_sqrt_6 = self.sqrt_6(opt_add_5)
        opt_div_7 = self.div_7(opt_sub_2, opt_sqrt_6)
        opt_mul_8 = self.mul_8(self.mul_8_w, opt_div_7)
        opt_add_9 = self.add_9(opt_mul_8, self.add_9_bias)
        return opt_add_9


class Module16(nn.Cell):
    """module of reader downstream"""
    def __init__(self, module15_0_matmul_0_weight_shape, module15_0_add_1_bias_shape, normmodule_0_mul_8_w_shape,
                 normmodule_0_add_9_bias_shape):
        """init function"""
        super(Module16, self).__init__()
        self.module15_0 = Module15(matmul_0_weight_shape=module15_0_matmul_0_weight_shape,
                                   add_1_bias_shape=module15_0_add_1_bias_shape)
        self.normmodule_0 = NormModule(mul_8_w_shape=normmodule_0_mul_8_w_shape,
                                       add_9_bias_shape=normmodule_0_add_9_bias_shape)
        self.matmul_0 = nn.MatMul()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, (8192, 1)).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        module15_0_opt = self.module15_0(x)
        normmodule_0_opt = self.normmodule_0(module15_0_opt)
        opt_matmul_0 = self.matmul_0(ops.Cast()(normmodule_0_opt, dst_type), ops.Cast()(self.matmul_0_w, dst_type))
        return ops.Cast()(opt_matmul_0, dst_type2)


class Module17(nn.Cell):
    """module of reader downstream"""
    def __init__(self, module15_0_matmul_0_weight_shape, module15_0_add_1_bias_shape, normmodule_0_mul_8_w_shape,
                 normmodule_0_add_9_bias_shape):
        """init function"""
        super(Module17, self).__init__()
        self.module15_0 = Module15(matmul_0_weight_shape=module15_0_matmul_0_weight_shape,
                                   add_1_bias_shape=module15_0_add_1_bias_shape)
        self.normmodule_0 = NormModule(mul_8_w_shape=normmodule_0_mul_8_w_shape,
                                       add_9_bias_shape=normmodule_0_add_9_bias_shape)
        self.matmul_0 = nn.MatMul()
        self.matmul_0_w = Parameter(Tensor(np.random.uniform(0, 1, (4096, 1)).astype(np.float32)), name=None)
        self.add_1 = P.Add()
        self.add_1_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)

    def construct(self, x):
        """construct function"""
        module15_0_opt = self.module15_0(x)
        normmodule_0_opt = self.normmodule_0(module15_0_opt)
        opt_matmul_0 = self.matmul_0(ops.Cast()(normmodule_0_opt, dst_type), ops.Cast()(self.matmul_0_w, dst_type))
        opt_add_1 = self.add_1(ops.Cast()(opt_matmul_0, dst_type2), self.add_1_bias)
        return opt_add_1


class Module5(nn.Cell):
    """module of reader downstream"""
    def __init__(self):
        """init function"""
        super(Module5, self).__init__()
        self.sub_0 = P.Sub()
        self.sub_0_bias = 1.0
        self.mul_1 = P.Mul()
        self.mul_1_w = 1.0000000150474662e+30

    def construct(self, x):
        """construct function"""
        opt_sub_0 = self.sub_0(self.sub_0_bias, x)
        opt_mul_1 = self.mul_1(opt_sub_0, self.mul_1_w)
        return opt_mul_1


class Module10(nn.Cell):
    """module of reader downstream"""
    def __init__(self):
        """init function"""
        super(Module10, self).__init__()
        self.squeeze_0 = P.Squeeze(2)
        self.module5_0 = Module5()
        self.sub_1 = P.Sub()

    def construct(self, x, x0):
        """construct function"""
        opt_squeeze_0 = self.squeeze_0(x)
        module5_0_opt = self.module5_0(x0)
        opt_sub_1 = self.sub_1(opt_squeeze_0, module5_0_opt)
        return opt_sub_1


class Reader_Downstream(nn.Cell):
    """Downstream model for reader"""
    def __init__(self):
        """init function"""
        super(Reader_Downstream, self).__init__()
        self.module16_0 = Module16(module15_0_matmul_0_weight_shape=(4096, 8192),
                                   module15_0_add_1_bias_shape=(8192,),
                                   normmodule_0_mul_8_w_shape=(8192,),
                                   normmodule_0_add_9_bias_shape=(8192,))
        self.add_74 = P.Add()
        self.add_74_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)
        self.module16_1 = Module16(module15_0_matmul_0_weight_shape=(4096, 8192),
                                   module15_0_add_1_bias_shape=(8192,),
                                   normmodule_0_mul_8_w_shape=(8192,),
                                   normmodule_0_add_9_bias_shape=(8192,))
        self.add_75 = P.Add()
        self.add_75_bias = Parameter(Tensor(np.random.uniform(0, 1, (1,)).astype(np.float32)), name=None)
        self.module17_0 = Module17(module15_0_matmul_0_weight_shape=(4096, 4096),
                                   module15_0_add_1_bias_shape=(4096,),
                                   normmodule_0_mul_8_w_shape=(4096,),
                                   normmodule_0_add_9_bias_shape=(4096,))
        self.module10_0 = Module10()
        self.module17_1 = Module17(module15_0_matmul_0_weight_shape=(4096, 4096),
                                   module15_0_add_1_bias_shape=(4096,),
                                   normmodule_0_mul_8_w_shape=(4096,),
                                   normmodule_0_add_9_bias_shape=(4096,))
        self.module10_1 = Module10()
        self.gather_6_input_weight = Tensor(np.array(0))
        self.gather_6_axis = 1
        self.gather_6 = P.Gather()
        self.dense_13 = nn.Dense(in_channels=4096, out_channels=4096, has_bias=True)
        self.relu_18 = nn.ReLU()
        self.normmodule_0 = NormModule(mul_8_w_shape=(4096,), add_9_bias_shape=(4096,))
        self.dense_73 = nn.Dense(in_channels=4096, out_channels=3, has_bias=True)

    def construct(self, x, x0, x1, x2):
        """construct function"""
        module16_0_opt = self.module16_0(x)
        opt_add_74 = self.add_74(module16_0_opt, self.add_74_bias)
        module16_1_opt = self.module16_1(x0)
        opt_add_75 = self.add_75(module16_1_opt, self.add_75_bias)
        module17_0_opt = self.module17_0(x1)
        opt_module10_0 = self.module10_0(module17_0_opt, x2)
        module17_1_opt = self.module17_1(x1)
        opt_module10_1 = self.module10_1(module17_1_opt, x2)
        opt_gather_6_axis = self.gather_6_axis
        opt_gather_6 = self.gather_6(x1, self.gather_6_input_weight, opt_gather_6_axis)
        opt_dense_13 = self.dense_13(opt_gather_6)
        opt_relu_18 = self.relu_18(opt_dense_13)
        normmodule_0_opt = self.normmodule_0(opt_relu_18)
        opt_dense_73 = self.dense_73(normmodule_0_opt)
        return opt_dense_73, opt_module10_0, opt_module10_1, opt_add_74, opt_add_75
