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
import pytest
import numpy as np
from mindspore import Tensor, context
from mindspore.ops.function.array_func import min_ext as min_
import mindspore.common.dtype as mstype
import mindspore as ms

from tests.st.utils.test_utils import to_cell_obj, compare
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def np_backward_func(np_input, axis, keep_dims, out_tuple, dout_tuple):
    value = out_tuple[0]
    dvalue = dout_tuple[0]
    if not keep_dims:
        value = np.expand_dims(value, axis)
        dvalue = np.expand_dims(dvalue, axis)
    return dvalue * np.equal(np_input, value)


def min_max_dim_case(op_func, np_func):
    x_np = np.array([[1, 20, 5], [67, 8, 9], [24, 15, 130]], np.float32)
    x = Tensor(x_np)
    net = to_cell_obj(op_func)
    axis = 1
    keepdims = False
    input_args = (x, axis, keepdims)
    # forward:
    output = net(*input_args)
    expect = np_func(x_np, axis, keepdims)
    compare(output, expect)
    # backward:
    output_grad = ms.grad(net, (0,))(*input_args)
    expect_grad = np_backward_func(x, axis, keepdims, expect,
                                   (np.ones_like(expect[0]), np.ones_like(expect[1])))
    assert np.allclose(output_grad.asnumpy(), expect_grad)


def min_max_dim_case_dyn(op_func, np_func, dyn_rank=False):
    axis = 1
    keepdims = False

    def func_dyn_case(x):
        return op_func(x, axis, keepdims)

    net = to_cell_obj(func_dyn_case)
    t1_np = np.array([[1, 20], [67, 8]], dtype=np.float32)
    if dyn_rank:
        net.set_inputs(Tensor(shape=None, dtype=mstype.float32))
        t2_np = np.array([[[1, 20, 5], [67, 8, 9]],
                          [[130, 24, 15], [16, 64, 32]]], dtype=np.float32)
    else:
        net.set_inputs(Tensor(shape=[None, None], dtype=mstype.float32))
        t2_np = np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]], dtype=np.float32)
    inputs = [t1_np, t2_np]
    for input_np in inputs:
        input_t = Tensor(input_np)
        # forward:
        output = net(input_t)
        expect = np_func(input_np, axis, keepdims)
        compare(output, expect)
        # backward:
        output_grad = ms.grad(net, (0,))(input_t)
        expect_grad = np_backward_func(input_np, axis, keepdims, expect,
                                       (np.ones_like(expect[0]), np.ones_like(expect[1])))
        assert np.allclose(output_grad.asnumpy(), expect_grad)


def np_min_dim(input_x, axis, keepdims):
    value = np.min(input_x, axis)
    index = np.argmin(input_x, axis).astype(np.int32)
    if keepdims:
        value = np.expand_dims(value, axis)
        index = np.expand_dims(index, axis)
    return value, index


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_min_dim(mode):
    """
    Feature: Test argmin_with_value op.
    Description: Test argmin_with_value.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case(min_, np_min_dim)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=1024.02)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_min_dim_dyn(mode):
    """
    Feature: Test argmin_with_value op.
    Description: Test argmin_with_value dynamic shape.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case_dyn(min_, np_min_dim)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'],
        level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.11)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_min_dim_dyn_rank(mode):
    """
    Feature: Test argmin_with_value op.
    Description: Test argmin_with_value dynamic rank.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=mode)
    min_max_dim_case_dyn(min_, np_min_dim, True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1024.01)
def test_min_dim_all_dynamic():
    """
    Feature: Test argmin_with_value op.
    Description: Test argmin_with_value with both input and axis are dynamic.
    Expectation: the result match with expected result.
    """
    t1 = Tensor(np.array([[1, 20], [67, 8]], dtype=np.float32))
    input_case1 = [t1, -1]
    t2 = Tensor(np.array([[[1, 20, 5], [67, 8, 9]], [[130, 24, 15], [16, 64, 32]]], dtype=np.float32))
    input_case2 = [t2, 0]
    TEST_OP(min_, [input_case1, input_case2],
            disable_mode=['GRAPH_MODE_GE', 'GRAPH_MODE_O0'],
            disable_case=['EmptyTensor'])
