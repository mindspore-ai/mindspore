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

from mindspore import ops as P
from mindspore import Tensor, nn
from mindspore import context
from mindspore.common import dtype as mstype


class ReshapeNet(nn.Cell):
    def __init__(self):
        super(ReshapeNet, self).__init__()
        self.reshape = P.Reshape()

    def construct(self, input_x, output_shape):
        return self.reshape(input_x, output_shape)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reshape_aicpu_dynamic_graph():
    """
    Feature: Test Reshape operation for dynamic shape in graph mode.
    Description: The input shape is dynamic, the output_shape is random tuple, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    context.set_context(mode=context.GRAPH_MODE)
    net = ReshapeNet()
    input_shape_dyn = Tensor(shape=(3, None, 5), dtype=mstype.float64)
    out_shape_dyn = Tensor(np.random.randint(0, 5, size=3))
    net.set_inputs(input_shape_dyn, out_shape_dyn)
    input_shape = (3, 4, 5)
    out_shape = [2, 6, 5]
    input_x = np.random.random(input_shape)
    output = net(Tensor(input_x), Tensor(out_shape))
    expect = input_x.reshape(out_shape)
    assert np.all(output.asnumpy() == expect)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_reshape_aicpu_dynamic_pynative():
    """
    Feature: Test Reshape operation for dynamic shape in pynative mode.
    Description: The input shape is dynamic, the output_shape is random tuple, value is a tensor.
    Expectation: Assert the result is equal the numpy result.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = ReshapeNet()
    input_shape_dyn = Tensor(shape=(3, None, 5), dtype=mstype.float64)
    out_shape_dyn = Tensor(np.random.randint(0, 5, size=3))
    net.set_inputs(input_shape_dyn, out_shape_dyn)
    input_shape = (3, 4, 5)
    out_shape = [2, 6, 5]
    input_x = np.random.random(input_shape)
    output = net(Tensor(input_x), Tensor(out_shape))
    expect = input_x.reshape(out_shape)
    assert np.all(output.asnumpy() == expect)
