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

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def hswish_grad_forward_func(y_grad, x):
    return ops.auto_generate.HSwishGrad()(y_grad, x)


@test_utils.run_with_cell
def hswish_grad_dyn_shape_func(y_grad, x):
    return ops.auto_generate.HSwishGrad()(y_grad, x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_grad_forward(mode):
    """
    Feature: Ops.
    Description: test op hswish_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([-1, -2, 0, 2, 1]).astype(np.float32))
    y_grad = ms.Tensor(np.array(
        [0.16666667, -0.16666667, 0.5, 1.1666666, 0.8333333]).astype(np.float32))
    expect_out = np.array([0.02777778, 0.02777778, 0.25, 1.361111, 0.6944444])
    out = hswish_grad_forward_func(y_grad, x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test hswish_grad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[-1, -2, 0, 2, 1]]).astype(np.float32))
    y_grad = ms.Tensor(np.array(
        [[-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667]])).astype(np.float32)
    expect_out = np.array(
        [[-0.05555556], [0.05555556], [0.], [1.9444443], [0.5555556]])
    nest_vmap = ops.vmap(ops.vmap(
        hswish_grad_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(y_grad, x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_grad_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of hswish_grad.
    Description: test dynamic tensor and dynamic scalar of hswish_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    y_grad_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = ms.Tensor(np.array([[-1, -2, 0, 2, 1]]).astype(np.float32))
    y_grad = ms.Tensor(np.array(
        [[0.16666667, -0.16666667, 0.5, 1.1666666, 0.8333333]]).astype(np.float32))
    test_cell = test_utils.to_cell_obj(hswish_grad_dyn_shape_func)
    test_cell.set_inputs(y_grad_dyn, x_dyn)
    out = test_cell(y_grad, x)
    expect = np.array([[0.02777778, 0.02777778, 0.25, 1.361111, 0.6944444]])
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[-2, 0, 2], [-1, 0, 2]]), ms.float32)
    y_grad1 = ms.Tensor(np.array(
        [[-0.33333334, 0., 1.6666667], [-0.33333333, 0., 1.6666667]]).astype(np.float32))
    out1 = test_cell(y_grad1, x1)
    expect1 = np.array(
        [[0.05555556, 0., 1.9444447], [-0.05555555, 0., 1.9444447]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_grad_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of hswish_grad.
    Description: test dynamic rank tensor of hswish_grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    y_grad_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x = ms.Tensor(np.array([[-1, -2, 0, 2, 1]]).astype(np.float32))
    y_grad = ms.Tensor(np.array(
        [[0.16666667, -0.16666667, 0.5, 1.1666666, 0.8333333]]).astype(np.float32))
    test_cell = test_utils.to_cell_obj(hswish_grad_dyn_shape_func)
    test_cell.set_inputs(y_grad_dyn, x_dyn)
    out = test_cell(y_grad, x)
    expect = np.array([[0.02777778, 0.02777778, 0.25, 1.361111, 0.6944444]])
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[-2, 0, 2], [-1, 0, 2]]), ms.float32)
    y_grad1 = ms.Tensor(np.array(
        [[-0.33333334, 0., 1.6666667], [-0.33333333, 0., 1.6666667]]).astype(np.float32))
    out1 = test_cell(y_grad1, x1)
    expect1 = np.array(
        [[0.05555556, 0., 1.9444447], [-0.05555555, 0., 1.9444447]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)
