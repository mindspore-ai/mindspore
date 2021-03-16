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
"""downstream Model for reranker"""

import numpy as np
from mindspore import nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class Rerank_Downstream(nn.Cell):
    """Downstream model for rerank"""
    def __init__(self):
        """init function"""
        super(Rerank_Downstream, self).__init__()
        self.dense_0 = nn.Dense(in_channels=4096, out_channels=8192, has_bias=True)
        self.relu_1 = nn.ReLU()
        self.reducemean_2 = P.ReduceMean(keep_dims=True)
        self.sub_3 = P.Sub()
        self.sub_4 = P.Sub()
        self.pow_5 = P.Pow()
        self.pow_5_input_weight = 2.0
        self.reducemean_6 = P.ReduceMean(keep_dims=True)
        self.add_7 = P.Add()
        self.add_7_bias = 9.999999960041972e-13
        self.sqrt_8 = P.Sqrt()
        self.div_9 = P.Div()
        self.mul_10 = P.Mul()
        self.mul_10_w = Parameter(Tensor(np.random.uniform(0, 1, (8192,)).astype(np.float32)), name=None)
        self.add_11 = P.Add()
        self.add_11_bias = Parameter(Tensor(np.random.uniform(0, 1, (8192,)).astype(np.float32)), name=None)
        self.dense_12 = nn.Dense(in_channels=8192, out_channels=2, has_bias=True)

    def construct(self, x):
        """construct function"""
        opt_dense_0 = self.dense_0(x)
        opt_relu_1 = self.relu_1(opt_dense_0)
        opt_reducemean_2 = self.reducemean_2(opt_relu_1, -1)
        opt_sub_3 = self.sub_3(opt_relu_1, opt_reducemean_2)
        opt_sub_4 = self.sub_4(opt_relu_1, opt_reducemean_2)
        opt_pow_5 = self.pow_5(opt_sub_3, self.pow_5_input_weight)
        opt_reducemean_6 = self.reducemean_6(opt_pow_5, -1)
        opt_add_7 = self.add_7(opt_reducemean_6, self.add_7_bias)
        opt_sqrt_8 = self.sqrt_8(opt_add_7)
        opt_div_9 = self.div_9(opt_sub_4, opt_sqrt_8)
        opt_mul_10 = self.mul_10(self.mul_10_w, opt_div_9)
        opt_add_11 = self.add_11(opt_mul_10, self.add_11_bias)
        opt_dense_12 = self.dense_12(opt_add_11)
        return opt_dense_12
