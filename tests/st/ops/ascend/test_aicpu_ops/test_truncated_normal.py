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
    def __init__(self, shape, dtype=mstype.float32, seed=0, seed2=0):
        super(Net, self).__init__()
        self.shape = shape
        self.dtype = dtype
        self.seed = seed
        self.seed2 = seed2
        self.truncated_normal = P.TruncatedNormal(dtype, seed, seed2)

    def construct(self):
        return self.truncated_normal(self.shape)


def test_net():
    """
    Feature: test TruncatedNormal op.
    Description: test TruncatedNormal op.
    Expectation: success.
    """
    shape = Tensor(np.array([2, 2]), mstype.int32)
    dtype = mstype.float32
    seed = 1
    seed2 = 2
    net = Net(shape, dtype, seed, seed2)
    output = net()
    assert output.shape == (2, 2)
