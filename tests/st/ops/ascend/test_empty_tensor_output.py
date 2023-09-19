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
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class TestReluNet(nn.Cell):
    def __init__(self):
        super(TestReluNet, self).__init__()
        self.relu = P.ReLU()

    def construct(self, x):
        return self.relu(x)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_empty_tensor_output():
    """
    Feature: test empty tensor output.
    Description: test op with empty tensor output, which should be skipped by framework.
    Expectation: the result match with numpy result.
    """
    relu_net = TestReluNet()
    x = Tensor(np.array([]).astype(np.float32))
    expect = np.array([]).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    output = relu_net(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    output = relu_net(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    x_dy = Tensor(shape=(None), dtype=mindspore.float32)
    relu_net.set_inputs(x_dy)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    output = relu_net(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    output = relu_net(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
