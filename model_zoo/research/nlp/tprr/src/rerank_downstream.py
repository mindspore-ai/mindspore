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


class BertLayerNorm(nn.Cell):
    """Layer norm for Bert"""
    def __init__(self, bln_weight=None, bln_bias=None, eps=1e-12):
        """init function"""
        super(BertLayerNorm, self).__init__()
        self.weight = bln_weight
        self.bias = bln_bias
        self.variance_epsilon = eps
        self.reduce_mean = P.ReduceMean(keep_dims=True)
        self.sub = P.Sub()
        self.pow = P.Pow()
        self.sqrt = P.Sqrt()
        self.div = P.Div()
        self.add = P.Add()
        self.mul = P.Mul()

    def construct(self, x):
        u = self.reduce_mean(x, -1)
        s = self.reduce_mean(self.pow(self.sub(x, u), 2.0), -1)
        x = self.div(self.sub(x, u), self.sqrt(self.add(s, self.variance_epsilon)))
        output = self.mul(self.weight, x)
        output = self.add(output, self.bias)

        return output


class Rerank_Downstream(nn.Cell):
    """Downstream model for rerank"""
    def __init__(self):
        """init function"""
        super(Rerank_Downstream, self).__init__()
        self.relu = nn.ReLU()
        self.linear_1 = nn.Dense(in_channels=4096, out_channels=8192, has_bias=True)
        self.linear_2 = nn.Dense(in_channels=8192, out_channels=2, has_bias=True)
        self.bln_1_weight = Parameter(Tensor(np.random.uniform(0, 1, (8192,)).astype(np.float32)), name=None)
        self.bln_1_bias = Parameter(Tensor(np.random.uniform(0, 1, (8192,)).astype(np.float32)), name=None)
        self.bln_1 = BertLayerNorm(bln_weight=self.bln_1_weight, bln_bias=self.bln_1_bias)

    def construct(self, cls_emd):
        """construct function"""
        output = self.linear_1(cls_emd)
        output = self.relu(output)
        output = self.bln_1(output)
        output = self.linear_2(output)
        return output
