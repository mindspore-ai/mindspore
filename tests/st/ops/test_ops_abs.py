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
import mindspore as ms
from mindspore import Tensor
from mindspore.ops.auto_generate import abs as abs_
from mindspore import ops
from mindspore import context
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.ops_binary_cases import ops_binary_cases, OpsBinaryCase


@test_utils.run_with_cell
def abs_forward_func(x):
    return abs_(x)


@test_utils.run_with_cell
def abs_backward_func(x):
    return ms.grad(abs_forward_func, (0))(x)


def abs_dyn_shape_func(x):
    return abs_(x)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.25)
@pytest.mark.parametrize('mode',
                         [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_forward(mode):
    """
    Feature: test abs operator
    Description: test abs forward by pyboost
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = Tensor([1.0, -2.0, -3.0], ms.float32)
    output = abs_forward_func(x)
    assert np.allclose(output.asnumpy(), [1.0, 2.0, 3.0])
    x = Tensor([1, 0, 0], ms.int8)
    output = abs_forward_func(x)
    assert np.allclose(output.asnumpy(), [1, 0, 0])


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.02)
@pytest.mark.parametrize('mode',
                         [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_backward(mode):
    """
    Feature: test abs operator
    Description: test abs backward by pyboost
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = Tensor([1.0, -2.0, -3.0], ms.float32)
    output = abs_backward_func(x)
    assert np.allclose(output.asnumpy(), [1.0, -1.0, -1.0])
    x = Tensor([1, 0, 0], ms.float32)
    output = abs_backward_func(x)
    assert np.allclose(output.asnumpy(), [1.0, 0, 0])


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=1024.01)
@pytest.mark.parametrize('mode',
                         [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_vmap(mode):
    """
    Feature: test vmap function.
    Description: test abs op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor([[1.0, -2.0, -3.0], [1.0, -2.0, -3.0]], ms.float32)
    abs_vmap = ops.vmap(abs_forward_func)
    output = abs_vmap(x)
    assert np.allclose(output.asnumpy(), [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode',
                         [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_dynamic_shape(mode):
    """
    Feature: test dynamic tensor of abs.
    Description: test dynamic tensor of abs.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = Tensor(shape=[None], dtype=ms.float32)
    x = Tensor([1.0, -2.0, -3.0], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(abs_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode',
                         [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_dynamic_rank(mode):
    """
    Feature: test dynamic tensor of abs.
    Description: test dynamic tensor of abs.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x = Tensor([1.0, -2.0, -3.0], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(abs_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


def ops_abs_binary_compare(input_binary_data, output_binary_data):
    output = abs_forward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[0], 1e-04, 1e-04)
    output = abs_backward_func(Tensor(input_binary_data[0]))
    assert np.allclose(output.asnumpy(), output_binary_data[1], 1e-04, 1e-04)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 500, 6), np.float32)],
                                output_info=[((1, 500, 6), np.float32), ((1, 500, 6), np.float32)],
                                extra_info='auto_drive'))
def ops_abs_binary_case1(input_binary_data=None, output_binary_data=None):
    ops_abs_binary_compare(input_binary_data, output_binary_data)


@ops_binary_cases(OpsBinaryCase(input_info=[((1, 2, 288, 64), np.float32)],
                                output_info=[((1, 2, 288, 64), np.float32), ((1, 2, 288, 64), np.float32)],
                                extra_info='auto_drive'))
def ops_abs_binary_case2(input_binary_data=None, output_binary_data=None):
    ops_abs_binary_compare(input_binary_data, output_binary_data)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='essential', mem_peak=1024.78)
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_abs_binary_cases(mode):
    """
    Feature: Ops
    Description: test op abs
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)

    ops_abs_binary_case1()
    ops_abs_binary_case2()
