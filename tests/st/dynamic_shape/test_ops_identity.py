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
def identity_forward_func(x):
    return ops.deepcopy(x)


@test_utils.run_with_cell
def identity_backward_func(x):
    return ops.grad(identity_forward_func, (0,))(x)


def identity_dyn_shape_func(x):
    return ops.deepcopy(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_identity_forward(mode):
    """
    Feature: Ops.
    Description: test op identity.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    output = identity_forward_func(input_x)
    expect = x
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_identity_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op identity.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = np.array([1.1, 2.5, -1.5]).astype(np.float32)
    input_x = ms.Tensor(x, ms.float32)
    grads = identity_backward_func(input_x)
    expect = np.array([1, 1, 1]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_identity_vmap(mode):
    """
    Feature: test vmap function.
    Description: test identity op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (0)
    np_x = np.array([[[1.1, 0.9], [2.2, 1.8]], [[4.6, 1.3], [2.4, 2.6]],
                     [[1.0, 1.0], [2.0, 2.7]], [[1.3, 1.7], [2.9, 2.8]],
                     [[1.1, 1.4], [2.6, 2.0]], [[1.2, 1.4], [2.0, 2.4]],
                     [[1.5, 1.4], [2.3, 2.0]], [[1.8, 1.0], [2.9, 2.0]]]).astype(np.float32)
    x = ms.Tensor(np_x)
    expect = np_x
    nest_vmap = ops.vmap(ops.vmap(identity_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    assert np.allclose(vmap_out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_identity_dynamic(mode):
    """
    Feature: test dynamic tensor of identity.
    Description: test dynamic tensor of identity.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(identity_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.arange(3).reshape(1, 3, 1, 1)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np_x
    assert np.allclose(output.asnumpy(), expect)
    np_x1 = np.arange(4).reshape(1, 1, 4, 1)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np_x1
    assert np.allclose(output1.asnumpy(), expect1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_identity_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of identity.
    Description: test dynamic rank tensor of identity.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(identity_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x = np.arange(3).reshape(1, 3)
    x = ms.Tensor(np_x, ms.float32)
    output = test_cell(x)
    expect = np_x
    assert np.allclose(output.asnumpy(), expect)
    np_x1 = np.arange(4).reshape(4, 1, 1, 1)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = np_x1
    assert np.allclose(output1.asnumpy(), expect1)
