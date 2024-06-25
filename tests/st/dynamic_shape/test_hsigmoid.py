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

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def hsigmoid_forward_func(x):
    return ops.HSigmoid()(x)


@test_utils.run_with_cell
def hsigmoid_backward_func(x):
    return ops.grad(hsigmoid_forward_func, (0,))(x)


@test_utils.run_with_cell
def hsigmoid_dyn_shape_func(x):
    return ops.HSigmoid()(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_hsigmoid_forward(mode):
    """
    Feature: Ops.
    Description: test op hsigmoid.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([1.0, 2.0, 3.0]).astype('float32')
    x = Tensor(np_array)
    out = hsigmoid_forward_func(x)
    expect = np.array([0.6666667, 0.8333333, 1.]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_hsigmoid_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op hsigmoid.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([1.0, 2.0, 3.0]).astype('float32')
    x = Tensor(np_array)
    grads = hsigmoid_backward_func(x)
    expect = np.array([0.16666667, 0.16666667, 0.]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_hsigmoid_vmap(mode):
    """
    Feature: test vmap function.
    Description: test hsigmoid op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([[0.5, 0.4, -0.3, -0.2]]).astype('float32')
    x = Tensor(np_array)
    nest_vmap = ops.vmap(ops.vmap(hsigmoid_forward_func, in_axes=0), in_axes=0)
    out = nest_vmap(x)
    expect = np.array(
        [[0.5833333, 0.56666666, 0.45000002, 0.46666667]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_hsigmoid_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of hsigmoid.
    Description: test dynamic tensor and dynamic scalar of hsigmoid.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    x = ms.Tensor(np.array([[1.0, 2.0, 3.0]]), ms.float32)
    test_cell = test_utils.to_cell_obj(hsigmoid_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([[0.6666667, 0.8333333, 1.]]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]), ms.float32)
    out1 = test_cell(x1)
    expect1 = np.array([[0.6666667, 0.8333333, 1., 1.]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_hsigmoid_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of hsigmoid.
    Description: test dynamic rank tensor of hsigmoid.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x = ms.Tensor(np.array([[1.0, 2.0, 3.0]]), ms.float32)
    test_cell = test_utils.to_cell_obj(hsigmoid_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    out = test_cell(x)
    expect = np.array([[0.6666667, 0.8333333, 1.]]).astype('float32')
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
    x1 = ms.Tensor(np.array([[1.0, 2.0, 3.0, 4.0]]), ms.float32)
    out1 = test_cell(x1)
    expect1 = np.array([[0.6666667, 0.8333333, 1., 1.]]).astype('float32')
    assert np.allclose(out1.asnumpy(), expect1, rtol=1e-4, atol=1e-4)
