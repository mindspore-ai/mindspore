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
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def sqrt_forward_func(x):
    return ops.sqrt(x)


@test_utils.run_with_cell
def sqrt_backward_func(x):
    return ops.grad(sqrt_forward_func, (0,))(x)


@test_utils.run_with_cell
def sqrt_dyn_shape_func(x):
    return ops.sqrt(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sqrt_forward(mode):
    """
    Feature: Ops.
    Description: test op sqrt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.2948122, 0.49372014]]).astype(np.float32))
    expect_out = np.array([[0.5429661, 0.7026522]]).astype(np.float32)
    out = sqrt_forward_func(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sqrt_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op sqrt.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.02595769, 0.25027096]]).astype(np.float32))
    expect_out = np.array([[3.1033945, 0.9994585]]).astype(np.float32)
    grads = sqrt_backward_func(x)
    assert np.allclose(grads.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sqrt_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sqrt op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[[0.21901467, 1.9256916]]]).astype(np.float32))
    expect_out = np.array([[[0.46801758]], [[1.3876953]]]).astype(np.float32)
    nest_vmap = ops.vmap(ops.vmap(
        sqrt_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sqrt_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of sqrt.
    Description: test dynamic tensor and dynamic scalar of sqrt.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(sqrt_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    x = ms.Tensor(np.array([[1.0, 4.0, 16.0]]).astype(np.float32))
    expect_out = np.array([[1.0, 2.0, 4.0]]).astype(np.float32)
    output = test_cell(x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    np_x1 = np.array([[1.0, 4.0, 9.0, 16.0]])
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect_out1 = np.array([[1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sqrt_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of sqrt.
    Description: test dynamic rank tensor of sqrt.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(sqrt_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    x = ms.Tensor(np.array([[1.0, 4.0, 16.0]]).astype(np.float32))
    expect_out = np.array([[1.0, 2.0, 4.0]]).astype(np.float32)
    output = test_cell(x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    np_x1 = np.array([[1.0, 4.0, 9.0, 16.0]])
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect_out1 = np.array([[1.0, 2.0, 3.0, 4.0]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)
