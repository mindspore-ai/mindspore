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
"""SE Module"""
from mindspore.ops import operations as P
from mindspore import Tensor

import mindspore.nn as nn
from mindspore.ops import functional as F
import numpy as np

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)

def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return nn.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class SELayer(nn.Cell):
    """SE Layer"""
    def __init__(self, out_channel, reduction=16):
        super(SELayer, self).__init__()
        self.se_global_pool = P.ReduceMean(keep_dims=False)
        self.se_dense_0 = _fc(out_channel, int(out_channel / 4))
        self.se_dense_1 = _fc(int(out_channel / 4), out_channel)
        self.se_sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.se_mul = P.Mul()

    def construct(self, out):
        out_se = out
        out = self.se_global_pool(out, (2, 3))
        out = self.se_dense_0(out)
        out = self.relu(out)
        out = self.se_dense_1(out)
        out = self.se_sigmoid(out)
        out = F.reshape(out, F.shape(out) + (1, 1))
        out = self.se_mul(out, out_se)
        return out
