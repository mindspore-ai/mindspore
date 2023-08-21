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
from mindspore.common import dtype as mstype
from mindspore.ops.operations import random_ops as P

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):
    def __init__(self, x, minval, maxval):
        super(Net, self).__init__()
        self.x = x
        self.minval = minval
        self.maxval = maxval
        self.uniform = P.Uniform(minval, maxval)

    def construct(self):
        return self.uniform(self.x)


def test_net():
    """
    Feature: test Uniform op.
    Description: test Uniform op.
    Expectation: success.
    """
    x = Tensor(np.random.randn(3, 4), mstype.float64)
    minval = 1.0
    maxval = 2.0
    net = Net(x, minval, maxval)
    output = net()
    assert output.shape == (3, 4)
