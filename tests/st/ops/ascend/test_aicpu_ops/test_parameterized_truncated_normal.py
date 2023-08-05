# Copyright 2023 Huawei Technologies Co., Ltd
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

import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, input_shape, input_mean, input_stdevs, input_min, input_max, seed=0, seed2=0):
        super(Net, self).__init__()
        self.shape = input_shape
        self.mean = input_mean
        self.stdevs = input_stdevs
        self.min = input_min
        self.max = input_max
        self.seed = seed
        self.seed2 = seed2
        self.parameterized_truncated_normal = P.ParameterizedTruncatedNormal(seed, seed2)

    def construct(self):
        return self.parameterized_truncated_normal(self.shape, self.mean, self.stdevs, self.min, self.max)


def test_net():
    """
    Feature: test ParameterizedTruncatedNormal op.
    Description: test ParameterizedTruncatedNormal op.
    Expectation: success.
    """
    input_shape = Tensor(np.array([2, 3]), mstype.int32)
    input_mean = Tensor(np.array([0]), mstype.float32)
    input_stdevs = Tensor(np.array([1]), mstype.float32)
    input_min = Tensor(np.array([-100]), mstype.float32)
    input_max = Tensor(np.array([100]), mstype.float32)
    seed = 1
    seed2 = 2
    net = Net(input_shape, input_mean, input_stdevs, input_min, input_max, seed, seed2)
    output = net()
    assert output.shape == (2, 3)
