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
def sqrt_grad_forward_func(dy, x):
    return ops.auto_generate.SqrtGrad()(dy, x)


@test_utils.run_with_cell
def sqrt_grad_dyn_shape_func(dy, x):
    return ops.auto_generate.SqrtGrad()(dy, x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sqrt_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op sqrt_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_np = np.array([[0.02595769, 0.25027096]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[0.0129776, 0.12512207]]).astype(np.float32)
    out = sqrt_grad_forward_func(dy, x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_sqrt_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test sqrt_grad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0, 0)
    x_np = np.array([[[0.21901467, 1.9256916]]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[[0.10955811, 0.9628906]]]).astype(np.float32)
    nest_vmap = ops.vmap(ops.vmap(
        sqrt_grad_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(dy, x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sqrt_grad_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of sqrt_grad.
    Description: test dynamic tensor and dynamic scalar of sqrt_grad.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(sqrt_grad_dyn_shape_func)
    test_cell.set_inputs(dy_dyn, x_dyn)
    x_np = np.array([[0.02595769, 0.25027096]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[0.0129776, 0.12512207]]).astype(np.float32)
    output = test_cell(dy, x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    x_np1 = np.array([[2.0, 5.0], [4.0, 3.0]]).astype(np.float32)
    dy1 = ms.Tensor(x_np1)
    x1 = ms.Tensor(x_np1 * x_np1)
    output1 = test_cell(dy1, x1)
    expect_out1 = np.array([[1.0, 2.5], [2.0, 1.5]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_sqrt_grad_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of sqrt_grad.
    Description: test dynamic rank tensor of sqrt_grad.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    dy_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(sqrt_grad_dyn_shape_func)
    test_cell.set_inputs(dy_dyn, x_dyn)
    x_np = np.array([[0.02595769, 0.25027096]]).astype(np.float32)
    dy = ms.Tensor(x_np)
    x = ms.Tensor(x_np * x_np)
    expect_out = np.array([[0.0129776, 0.12512207]]).astype(np.float32)
    output = test_cell(dy, x)
    assert np.allclose(output.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)
    x_np1 = np.array([[2.0, 5.0], [4.0, 3.0]]).astype(np.float32)
    dy1 = ms.Tensor(x_np1)
    x1 = ms.Tensor(x_np1 * x_np1)
    output1 = test_cell(dy1, x1)
    expect_out1 = np.array([[1.0, 2.5], [2.0, 1.5]]).astype(np.float32)
    assert np.allclose(output1.asnumpy(), expect_out1, rtol=1e-4, atol=1e-4)
