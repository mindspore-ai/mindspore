# Copyright 2022 Huawei Technologies Co., Ltd
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
import pytest

import mindspore as ms
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.op = P.DivNoNan()

    def construct(self, x, y):
        return self.op(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_divnonan_dshape():
    """
    Feature: Test divnonan dynamic shape.
    Description: Test divnonan dynamic shape.
    Expectation: Success.
    """
    net = Net()
    input_x_dyn = Tensor(shape=[3, None], dtype=ms.float32)
    input_y_dyn = Tensor(shape=[None, 1], dtype=ms.float32)
    net.set_inputs(input_x_dyn, input_y_dyn)
    input_x = Tensor(np.random.random(([3, 10])), dtype=ms.float32)
    input_y = Tensor(np.random.random(([3, 1])), dtype=ms.float32)
    output = net(input_x, input_y)
    expect_shape = (3, 10)
    assert output.asnumpy().shape == expect_shape
