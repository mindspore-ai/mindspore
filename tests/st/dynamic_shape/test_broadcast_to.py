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
def broadcast_to_forward_func(x, shape):
    return ops.auto_generate.broadcast_to(x, shape)


@test_utils.run_with_cell
def broadcast_to_backward_func(x, shape):
    return ops.grad(broadcast_to_forward_func, (0,))(x, shape)


@test_utils.run_with_cell
def broadcast_to_dyn_shape_func(x, shape):
    return ops.auto_generate.broadcast_to(x, shape)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_broadcast_to_forward(mode):
    """
    Feature: Ops.
    Description: test op gelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    shape = (3, 5, 7, 4, 5, 6)
    x_np = np.arange(20).reshape((4, 5, 1)).astype(np.int32)
    x = Tensor(x_np)
    out = broadcast_to_forward_func(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_broadcast_to_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    shape = (2, 3, 2)
    x_np = np.arange(6).reshape((2, 3, 1)).astype(np.float32)
    x = Tensor(x_np)
    grads = broadcast_to_backward_func(x, shape)
    expect = np.array([[[2.], [2.], [2.]], [[2.], [2.], [2.]]]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_broadcast_to_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (-1, None)
    shape = (4, 5, 6)
    x_np = np.arange(80).reshape((4, 5, 1, 2, 2)).astype(np.float32)
    x = Tensor(x_np)
    nest_vmap = ops.vmap(ops.vmap(broadcast_to_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x, shape)
    np_shape = (4, 5, 6, 2, 2)
    expect = np.broadcast_to(x_np, np_shape)
    expect = expect.transpose((4, 3, 0, 1, 2))
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_broadcast_to_dynamic(mode):
    """
    Feature: test dynamic tensor and dynamic scalar of gelu.
    Description: test dynamic tensor and dynamic scalar of gelu.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, 5, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(broadcast_to_dyn_shape_func)
    shape = (4, 5, 6)
    x_np = np.arange(20).reshape((4, 5, 1)).astype(np.float32)
    x = Tensor(x_np)
    test_cell.set_inputs(x_dyn, shape)
    output = test_cell(x, shape)
    expect = np.broadcast_to(x_np, shape)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
