# Copyright 2020 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import numpy as np
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
import pytest


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sqrt = P.Sqrt()

    def construct(self, x):
        return self.sqrt(x)


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_x, output_grad):
        return self.grad(self.network)(input_x, output_grad)


def sqrt_forward_test(ms_type, np_type):
    input_x = Tensor(np.array([4.0, 9.0, 16.0, 25.0, 36.0]), ms_type)
    sqrt = Net()
    output = sqrt(input_x)
    expected = np.array([2.0, 3.0, 4.0, 5.0, 6.0]).astype(np_type)
    if ms_type == mindspore.bfloat16:
        output_np = output.float().asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=3)
    else:
        output_np = output.asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=6)


def sqrt_backward_test(ms_type, np_type):
    x = Tensor(np.array([1.0, 1.0]), ms_type)
    sens = Tensor(np.array([0.1, 0.1]), ms_type)
    net = Grad(Net())
    output = net(x, sens)
    expected = np.array([0.05, 0.05]).astype(np_type)
    if ms_type == mindspore.bfloat16:
        output_np = output[0].float().asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=3)
    else:
        output_np = output[0].asnumpy()
        np.testing.assert_array_almost_equal(output_np, expected, decimal=6)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sqrt_forward_fp32(mode):
    """
    Feature: test sqrt forward with mstype.float32, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    sqrt_forward_test(mindspore.float32, np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sqrt_forward_bf16(mode):
    """
    Feature: test sqrt forward with mstype.bfloat16, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    sqrt_forward_test(mindspore.bfloat16, np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sqrt_backward_fp32(mode):
    """
    Feature: test sqrt backward with mstype.float32, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    sqrt_backward_test(mindspore.float32, np.float32)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sqrt_backward_bf16(mode):
    """
    Feature: test sqrt backward with mstype.bfloat16, on mode GRAPH & PYNATIVE
    Description: test inputs using given mindspore type and data type
    Expectation: the result match with the expected result
    """
    context.set_context(mode=mode, device_target="Ascend")
    sqrt_backward_test(mindspore.bfloat16, np.float32)
