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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore.ops import minimum
from mindspore import ops

from tests.st.utils.test_utils import to_cell_obj, compare
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP

def np_backward_func(np_x, np_y, dout, is_minimum):
    input_equal = np.equal(np_x, np_y)
    dout_scale = np.where(input_equal, dout/2.0, dout)
    zeros = np.zeros_like(dout_scale)
    dx = np.where(np_x > np_y, zeros, dout_scale)
    dy = np.where(np_x < np_y, zeros, dout_scale)
    if not is_minimum:
        dy, dx = dx, dy
    return dx, dy


def minimum_maximum_case(op_func, np_func, is_minimum=True):
    x_np = np.array([[1, 20, 5], [67, 8, 9], [24, 15, 130]], np.float32)
    x = Tensor(x_np)
    y_np = np.array([[2, 23, 5], [66, 8, 8], [24, 16, 120]], np.float32)
    y = Tensor(y_np)
    net = to_cell_obj(op_func)
    input_args = (x, y)
    # forward:
    output = net(*input_args)
    expect = np_func(x_np, y_np)
    compare(output, expect)
    # backward:
    output_grad = ops.grad(net, (0, 1))(*input_args)
    expect_grad = np_backward_func(x_np, y_np, np.ones_like(expect), is_minimum)
    compare(output_grad, expect_grad)


def minimum_maximum_case_vmap(op_func):
    def func_vmap_case(x, y):
        return op_func(x, y)

    x_batched = np.array([[5., 3., 4.], [2., 4., 3.], [3., 1., 4.]], dtype=np.float32)
    y_batched = np.array([[2., 4., 3.], [3., 1., 4.], [5., 3., 4.]], dtype=np.float32)
    output_vmap = ops.vmap(func_vmap_case, in_axes=0)(Tensor(x_batched), Tensor(y_batched))
    value_batched = []
    for x, y in zip(x_batched, y_batched):
        value = func_vmap_case(Tensor(x), Tensor(y))
        value_batched.append(value.asnumpy())
    expect = np.stack(value_batched)
    compare(output_vmap, expect)


def minimum_maximum_case_all_dyn(op_func):
    t1_x = Tensor(np.array([[1, 20], [67, 8]], dtype=np.float32))
    t1_y = Tensor(np.array([[2, 10], [67, 9]], dtype=np.float32))
    input_case1 = [t1_x, t1_y]
    t2_x = Tensor(np.array([[[1, 20, 5], [67, 8, 9]], [[130, 24, 15], [16, 64, 32]]], dtype=np.float32))
    t2_y = Tensor(np.array([[[0, 20, 6], [69, 10, 9]], [[120, 20, 14], [16, 64, 36]]], dtype=np.float32))
    input_case2 = [t2_x, t2_y]
    TEST_OP(op_func, [input_case1, input_case2], '', disable_yaml_check=True)


def np_minimum(input_x, input_y):
    return np.minimum(input_x, input_y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_minimum(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case(minimum, np_minimum)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_minimum_vmap(mode):
    """
    Feature: Test minimum op.
    Description: Test minimum vmap.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    minimum_maximum_case_vmap(minimum)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_minimum_all_dynamic():
    """
    Feature: Test minimum op.
    Description: Test minimum with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    minimum_maximum_case_all_dyn(minimum)
