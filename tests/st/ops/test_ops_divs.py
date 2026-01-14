# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor, context
from mindspore import mint, nn


class DivsNetTensorTensor(nn.Cell):
    def __init__(self):
        super(DivsNetTensorTensor, self).__init__()
        self.x = Tensor([5, 6], dtype=ms.int16)
        self.y = Tensor([2, 3], dtype=ms.int16)

    def construct(self):
        return mint.div(self.x, self.y)


class DivsNetTensorScalar(nn.Cell):
    def __init__(self):
        super(DivsNetTensorScalar, self).__init__()
        self.x = Tensor([5, 6], dtype=ms.int16)
        self.y = 3

    def construct(self):
        return mint.div(self.x, self.y)


@test_utils.run_with_cell
def divs_forward_func(x, y):
    return mint.div(x, y)


@test_utils.run_with_cell
def divs_backward_func(x, y):
    return ms.grad(divs_forward_func, (0,))(x, y)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def get_np_golden(input_x, other, grad):
    input_array = input_x.asnumpy()
    grad_array = grad.asnumpy()
    return input_array / other, grad_array / other


def compare_result(actual, expected):
    diff = abs(actual.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize("mode", ["KBK"])
def test_divs_infer_value(mode):
    """
    Feature: Divs for infer value
    Description: test ops divs
    Expectation: expect correct result.
    """
    set_mode(mode)

    net1 = DivsNetTensorTensor()
    output1 = net1()
    np.allclose(output1.asnumpy(), np.array([2.5, 2]))

    net2 = DivsNetTensorScalar()
    output2 = net2()
    np.allclose(output2.asnumpy(), np.array([1.66666, 2]), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_divs_static(mode):
    """
    Feature: Divs
    Description: test ops divs
    Expectation: expect correct result.
    """
    set_mode(mode)

    input_x = Tensor(np.random.randn(1, 2, 4, 5, 6).astype(np.float32))
    other = 2.
    grad = Tensor(np.ones((1, 2, 4, 5, 6)).astype(np.float32))
    golden_output, golden_gradient = get_np_golden(input_x, other, grad)

    forward_out = divs_forward_func(input_x, other)
    backward_out = divs_backward_func(input_x, other)
    compare_result(forward_out, golden_output)
    compare_result(backward_out, golden_gradient)


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_divs_dynamic():
    """
    Feature: Divs
    Description: test op divs dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.
    # test case 1
    x = Tensor(np.random.randn(1, 4, 5, 5), dtype=ms.float32)
    y = 2
    input_case1 = [x, y]
    # test case 2
    x = Tensor(np.random.randn(20, 15), dtype=ms.float32)
    y = 15
    input_case2 = [x, y]
    TEST_OP(
        divs_forward_func,
        [input_case1, input_case2],
        disable_mode=["GRAPH_MODE_GE"]
    )
