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


class DivModsNetTensorTensor(nn.Cell):
    def __init__(self):
        super(DivModsNetTensorTensor, self).__init__()
        self.x = Tensor([2, -3, 2.4, 4], dtype=ms.float16)
        self.y = Tensor([1, 2, 2, 4], dtype=ms.float16)

    def construct(self):
        return mint.div(self.x, self.y, rounding_mode='trunc')

class DivModsNetTensorTensor2(nn.Cell):
    def __init__(self):
        super(DivModsNetTensorTensor2, self).__init__()
        self.x = Tensor([2, -3, 2.4, 4], dtype=ms.float16)
        self.y = Tensor([1, 2, 2, 4], dtype=ms.float16)

    def construct(self):
        return mint.div(self.x, self.y, rounding_mode='floor')


class DivModsNetTensorScalar(nn.Cell):
    def __init__(self):
        super(DivModsNetTensorScalar, self).__init__()
        self.x = Tensor([2, -3, 2.4, 4], dtype=ms.float16)
        self.y = 2

    def construct(self):
        return mint.div(self.x, self.y, rounding_mode='trunc')


class DivModsNetTensorScalar2(nn.Cell):
    def __init__(self):
        super(DivModsNetTensorScalar2, self).__init__()
        self.x = Tensor([2, -3, 2.4, 4], dtype=ms.float16)
        self.y = 2

    def construct(self):
        return mint.div(self.x, self.y, rounding_mode='floor')


@test_utils.run_with_cell
def divmods_none_forward_func(x, y):
    return mint.div(x, y, rounding_mode=None)


@test_utils.run_with_cell
def divmods_none_backward_func(x, y):
    return ms.grad(divmods_none_forward_func, (0,))(x, y)


@test_utils.run_with_cell
def divmods_trunc_forward_func(x, y):
    return mint.div(x, y, rounding_mode="trunc")


@test_utils.run_with_cell
def divmods_trunc_backward_func(x, y):
    return ms.grad(divmods_trunc_forward_func, (0,))(x, y)


@test_utils.run_with_cell
def divmods_floor_forward_func(x, y):
    return mint.div(x, y, rounding_mode="floor")


@test_utils.run_with_cell
def divmods_floor_backward_func(x, y):
    return ms.grad(divmods_floor_forward_func, (0,))(x, y)


def set_mode(mode):
    """
    set mode
    """
    if mode == "KBK":
        context.set_context(mode=context.GRAPH_MODE, jit_config={"jit_level": "O0"})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


def compare_result(actual, expected):
    diff = abs(actual.asnumpy() - expected)
    error = np.ones(shape=expected.shape) * 1.0e-4
    assert np.all(diff < error)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("mode", ["KBK"])
def test_divmods_infer_value(mode):
    """
    Feature: Divmods for infer value
    Description: test ops divs
    Expectation: expect correct result.
    """
    set_mode(mode)

    net1 = DivModsNetTensorTensor()
    out1 = net1()
    np.allclose(out1.asnumpy(), np.array([2, -1, 1, 1]), 0.0001, 0.0001)

    net2 = DivModsNetTensorTensor2()
    out2 = net2()
    np.allclose(out2.asnumpy(), np.array([2, -1, 1, 1]), 0.0001, 0.0001)

    net3 = DivModsNetTensorScalar()
    out3 = net3()
    np.allclose(out3.asnumpy(), np.array([1, -1, 1, 2]), 0.0001, 0.0001)

    net4 = DivModsNetTensorScalar2()
    out4 = net4()
    np.allclose(out4.asnumpy(), np.array([1, -2, 1, 2]), 0.0001, 0.0001)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential',
        mem_peak=2048.00)
@pytest.mark.parametrize("mode", ["KBK", "PYBOOST"])
def test_divmods_static(mode):
    """
    Feature: DivMods
    Description: test ops divmods
    Expectation: expect correct result.
    """
    set_mode(mode)

    input_x = np.array([[0.03580964, -0.583409],
                        [2.220165, 0.50147223]]).astype(np.float32)
    input_x = Tensor(input_x)
    other = 3.

    # mode none
    expect_forward = np.array([[0.01193655, -0.19446968],
                               [0.740055, 0.16715741]]).astype(np.float32)
    forward = divmods_none_forward_func(input_x, other)
    compare_result(forward, expect_forward)

    expect_backward = np.array([[0.33333334, 0.33333334],
                                [0.33333334, 0.33333334]]).astype(np.float32)
    backward = divmods_none_backward_func(input_x, other)
    compare_result(backward, expect_backward)

    # mode floor
    expect_forward = np.array([[0., -1.],
                               [0., 0.]]).astype(np.float32)
    forward = divmods_floor_forward_func(input_x, other)
    compare_result(forward, expect_forward)

    expect_backward = np.array([[0., 0.],
                                [0., 0.]]).astype(np.float32)
    backward = divmods_floor_backward_func(input_x, other)
    compare_result(backward, expect_backward)

    # mode trunc
    expect_forward = np.array([[0., -0.],
                               [0., 0.]]).astype(np.float32)
    forward = divmods_trunc_forward_func(input_x, other)
    compare_result(forward, expect_forward)

    expect_backward = np.array([[0., 0.],
                                [0., 0.]]).astype(np.float32)
    backward = divmods_trunc_backward_func(input_x, other)
    compare_result(backward, expect_backward)


def generate_inputs():
    input_shape = (1, 2, 3, 4, 5, 6)
    input_x = Tensor(np.random.randn(*input_shape), dtype=ms.float32)
    other = 14
    inputs_1 = [input_x, other]

    input_shape = (20, 10, 10)
    input_x = Tensor(np.random.randn(*input_shape), dtype=ms.float32)
    other = 5
    inputs_2 = [input_x, other]
    return [inputs_1, inputs_2]


@arg_mark(
    plat_marks=["platform_ascend"],
    level_mark="level1",
    card_mark="onecard",
    essential_mark="unessential", mem_peak=2048.01
)
def test_divmods_dynamic():
    """
    Feature: DivMods
    Description: test op divmods dynamic shape
    Expectation: expect correct result.
    """
    context.set_context(
        runtime_num_threads=1
    )  # multi-threads have none-initialized bug now.

    TEST_OP(
        divmods_none_forward_func,
        generate_inputs(),
        disable_mode=["GRAPH_MODE_GE"]
    )

    TEST_OP(
        divmods_trunc_forward_func,
        generate_inputs(),
        disable_mode=["GRAPH_MODE_GE"]
    )

    TEST_OP(
        divmods_floor_forward_func,
        generate_inputs(),
        disable_mode=["GRAPH_MODE_GE"]
    )
