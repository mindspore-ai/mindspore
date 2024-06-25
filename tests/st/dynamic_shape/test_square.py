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

from mindspore import ops
import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def square_forward_func(x):
    return ops.square(x)


@test_utils.run_with_cell
def square_backward_func(x):
    return ops.grad(square_forward_func, (0,))(x)


@test_utils.run_with_cell
def square_dyn_shape_func(x):
    return ops.square(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_square_forward(mode):
    """
    Feature: Ops.
    Description: test op square.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1.0, 2.0, 4.0]]).astype(np.float32))
    expect_out = np.array([[1.0, 4.0, 16.0]]).astype(np.float32)
    out = square_forward_func(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_square_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op square.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1.0, 2.0, 4.0]]).astype(np.float32))
    expect_out = np.array([[2.0, 4.0, 8.0]]).astype(np.float32)
    grads = square_backward_func(x)
    assert np.allclose(grads.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_square_vmap(mode):
    """
    Feature: test vmap function.
    Description: test square op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[[1.0, 2.0, 4.0]]]).astype(np.float32))
    expect_out = np.array([[[1.0]], [[4.0]], [[16.0]]]).astype(np.float32)
    nest_vmap = ops.vmap(ops.vmap(
        square_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_square_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of square.
    Description: test dynamic tensor and dynamic scalar of square.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    x = ms.Tensor(np.array([[1.0, 2.0, 4.0]]).astype(np.float32))
    expect_out = np.array([[1.0, 4.0, 16.0]]).astype(np.float32)
    output = test_cell(x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    np_x1 = np.array([[1.0, 2.0, 3.0, 4.0]])
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect_out1 = np.array([[1.0, 4.0, 9.0, 16.0]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_square_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of square.
    Description: test dynamic rank tensor of square.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(square_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    x = ms.Tensor(np.array([[1.0, 2.0, 4.0]]).astype(np.float32))
    expect_out = np.array([[1.0, 4.0, 16.0]]).astype(np.float32)
    output = test_cell(x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    np_x1 = np.array([[1.0, 2.0, 3.0, 4.0]])
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect_out1 = np.array([[1.0, 4.0, 9.0, 16.0]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)
