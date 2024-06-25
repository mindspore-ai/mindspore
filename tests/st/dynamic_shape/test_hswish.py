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
def hswish_forward_func(x):
    return ops.HSwish()(x)


@test_utils.run_with_cell
def hswish_backward_func(x):
    return ops.grad(hswish_forward_func, (0,))(x)


@test_utils.run_with_cell
def hswish_dyn_shape_func(x):
    return ops.HSwish()(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_forward(mode):
    """
    Feature: Ops.
    Description: test op hswish.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([-1, -2, 0, 2, 1]).astype(np.float32))
    expect_out = np.array([-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667])
    out = hswish_forward_func(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op hswish.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([-1, -2, 0, 2, 1]).astype(np.float32))
    expect_out = np.array([0.16666667, -0.16666667, 0.5, 1.1666666, 0.8333333])
    grads = hswish_backward_func(x)
    assert np.allclose(grads.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_vmap(mode):
    """
    Feature: test vmap function.
    Description: test hswish op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.array([[[-1, -2, 0, 2, 1]]]).astype(np.float32))
    expect_out = np.array(
        [[[-0.33333334]], [[-0.33333334]], [[0.]], [[1.6666666]], [[0.6666667]]])
    nest_vmap = ops.vmap(ops.vmap(
        hswish_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    assert np.allclose(out.asnumpy(), expect_out, 1e-04, 1e-04)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of hswish.
    Description: test dynamic tensor and dynamic scalar of hswish.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = ms.Tensor(np.array([[-1, -2, 0, 2, 1]]).astype(np.float32))
    test_cell = test_utils.to_cell_obj(hswish_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([[-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667]])
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[-2, 0, 2], [-1, 0, 2]]), ms.float32)
    out1 = test_cell(x1)
    expect1 = np.array(
        [[-0.33333334, 0., 1.6666667], [-0.33333333, 0., 1.6666667]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hswish_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of hswish.
    Description: test dynamic rank tensor of hswish.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = ms.Tensor(np.array([[-1, -2, 0, 2, 1]]).astype(np.float32))
    test_cell = test_utils.to_cell_obj(hswish_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([[-0.33333334, -0.33333334, 0., 1.6666666, 0.6666667]])
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[-2, 0, 2], [-1, 0, 2]]), ms.float32)
    out1 = test_cell(x1)
    expect1 = np.array(
        [[-0.33333334, 0., 1.6666667], [-0.33333333, 0., 1.6666667]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)
