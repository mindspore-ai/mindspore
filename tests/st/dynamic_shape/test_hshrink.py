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
from tests.st.utils import test_utils

from mindspore import ops
from mindspore import Tensor
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def hshrink_forward_func(input_x, lambd):
    return ops.HShrink(lambd)(input_x)


@test_utils.run_with_cell
def hshrink_backward_func(input_x, lambd):
    return ops.grad(hshrink_forward_func, (0,))(input_x, lambd)


@test_utils.run_with_cell
def hshrink_dyn_shape_func(input_x, lambd):
    return ops.HShrink(lambd)(input_x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hshrink_forward(context_mode):
    """
    Feature: Ops.
    Description: test op hshrink.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array(
        [[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]).astype('float32')
    input_x = Tensor(np_array)
    lambd = 0.5
    out = hshrink_forward_func(input_x, lambd)
    expect = np.array([[0., 1., 2.], [0., 0., -2.1233]]).astype('float32')
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hshrink_backward(context_mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op hshrink.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    np_array = np.array(
        [[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]).astype('float32')
    input_x = Tensor(np_array)
    lambd = 0.5
    grads = hshrink_backward_func(input_x, lambd)
    expect = np.array([[0., 1., 1.], [0., 0., 1]]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_hshrink_vmap(context_mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    in_axes = (0, None)
    ms.context.set_context(mode=context_mode)
    np_array = np.array(
        [[[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]]).astype('float32')
    input_x = Tensor(np_array)
    lambd = 0.5
    nest_vmap = ops.vmap(ops.vmap(hshrink_forward_func, in_axes=in_axes), in_axes=in_axes)
    out = nest_vmap(input_x, lambd)
    expect = np.array(
        [[[0, 1., 2.], [0., 0., -2.1233]]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hshrink_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of hshrink.
    Description: test dynamic tensor and dynamic scalar of hshrink.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    np_array = np.array(
        [[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]).astype('float32')
    input_x = Tensor(np_array)
    lambd = 0.5
    test_cell = test_utils.to_cell_obj(hshrink_dyn_shape_func)
    test_cell.set_inputs(x_dyn, lambd)
    out = test_cell(input_x, lambd)
    expect = np.array([[0., 1., 2.], [0., 0., -2.1233]]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    input_x1 = Tensor(np.array(
        [[0.25, 0.4, 0.01, 1], [0.1533, 1.0776, -2.1233, 0.2]]).astype('float32'))
    lambd = 0.5
    test_cell = test_utils.to_cell_obj(hshrink_dyn_shape_func)
    test_cell.set_inputs(x_dyn, lambd)
    out1 = test_cell(input_x1, lambd)
    expect1 = np.array(
        [[0., 0., 0., 1.], [0., 1.0776, -2.1233, 0]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hshrink_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of hshrink.
    Description: test dynamic rank tensor of hshrink.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    np_array = np.array(
        [[0.5, 1, 2.0], [0.0533, 0.0776, -2.1233]]).astype('float32')
    input_x = Tensor(np_array)
    lambd = 0.5
    test_cell = test_utils.to_cell_obj(hshrink_dyn_shape_func)
    test_cell.set_inputs(x_dyn, lambd)
    out = test_cell(input_x, lambd)
    expect = np.array([[0., 1., 2.], [0., 0., -2.1233]]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    input_x1 = Tensor(np.array(
        [[0.25, 0.4, 0.01, 1], [0.1533, 1.0776, -2.1233, 0.2]]).astype('float32'))
    lambd = 0.5
    test_cell = test_utils.to_cell_obj(hshrink_dyn_shape_func)
    test_cell.set_inputs(x_dyn, lambd)
    out1 = test_cell(input_x1, lambd)
    expect1 = np.array(
        [[0., 0., 0., 1.], [0., 1.0776, -2.1233, 0]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)
