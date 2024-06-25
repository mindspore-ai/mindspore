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
from tests.st.utils import test_utils

from  mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def celu_forward_func(x, alpha=1.0):
    return ops.CeLU(alpha)(x)


@test_utils.run_with_cell
def celu_backward_func(x, alpha=1.0):
    return ops.grad(celu_forward_func, (0,))(x, alpha)


def celu_dyn_shape_func(x, alpha=1.0):
    return ops.CeLU(alpha)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_with_On
def test_celu_forward(mode):
    """
    Feature: Ops.
    Description: test op celu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    error = 1e-4
    expect = np.array([-0.86468, -0.63212, 1., 2.]).astype(np.float32)
    x = np.array([-2.0, -1.0, 1.0, 2.0]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    output = celu_forward_func(input_x, 1.0)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_with_On
def test_celu_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op celu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    error = 1e-4
    expect = np.array([1, 1, 0.22313])
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    grads = celu_backward_func(input_x, 1.0)
    np.testing.assert_allclose(grads.asnumpy(), expect, rtol=error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_with_On
def test_celu_vmap(mode):
    """
    Feature: test vmap function.
    Description: test celu op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    error = 1e-4
    in_axes = (0)
    x = ms.Tensor(np.array([[[-2.0, -1.0, 1.0, 2.0]]]).astype(np.float32))
    nest_vmap = ops.vmap(celu_forward_func, in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    expect = np.array([[[-0.86468, -0.63212, 1., 2.]]]).astype(np.float32)
    np.testing.assert_allclose(vmap_out.asnumpy(), expect, rtol=error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_celu_dynamic(mode):
    """
    Feature: test dynamic tensor of celu.
    Description: test dynamic tensor of celu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    error = 1e-4
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(celu_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.array([[-2.0, -1.0], [1.0, 2.0]]).astype(np.float32)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np.array([[-0.86468, -0.63212], [1., 2.]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
    np_x1 = np.array([[-2.0, -1.0, 1.0, 2.0]]).astype(np.float32)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np.array([[-0.86468, -0.63212, 1., 2.]]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=error)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_celu_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of celu.
    Description: test dynamic rank tensor of celu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    error = 1e-4
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(celu_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.array([[-2.0, -1.0], [1.0, 2.0]]).astype(np.float32)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np.array([[-0.86468, -0.63212], [1., 2.]]).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=error)
    np_x1 = np.array([-2.0, -1.0, 1.0, 2.0]).astype(np.float32)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np.array([-0.86468, -0.63212, 1., 2.]).astype(np.float32)
    np.testing.assert_allclose(output1.asnumpy(), expect1, rtol=error)
