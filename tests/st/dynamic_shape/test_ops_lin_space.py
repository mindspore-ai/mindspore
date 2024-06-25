# Copyright 2020 Huawei Technologies Co., Ltd
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
def lin_space_forward_func(start, stop, num=5):
    return ops.LinSpace()(start, stop, num)


@test_utils.run_with_cell
def lin_space_backward_func(start, stop, num=5):
    return ops.grad(lin_space_forward_func, (0,))(start, stop, num)


def lin_space_dyn_shape_func(start, stop, num=5):
    return ops.LinSpace()(start, stop, num)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_lin_space_forward(mode):
    """
    Feature: Ops.
    Description: test op LinSpace.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    start, stop, num = 5, 25, 5
    output = lin_space_forward_func(ms.Tensor(start, ms.float32), ms.Tensor(stop, ms.float32), num)
    expect = np.linspace(start, stop, num, axis=-1)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_lin_space_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op LinSpace.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    start, stop, num = 5, 25, 5
    grads = lin_space_forward_func(ms.Tensor(start, ms.float32), ms.Tensor(stop, ms.float32), num)
    expect = np.array([5., 10., 15., 20., 25.]).astype(np.float32)
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_lin_space_vmap(mode):
    """
    Feature: test vmap function.
    Description: test LinSpace op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np.random.seed(0)
    start_np = np.random.randn(5, 4)
    stop_np = np.random.randn(4, 5)
    num_np = 5
    start = ms.Tensor(start_np, dtype=ms.float32)
    stop = ms.Tensor(stop_np, dtype=ms.float32)
    result_ms = ops.vmap(ops.vmap(lin_space_forward_func, (0, 0)), (1, 0))(start, stop)
    start_np = np.moveaxis(start_np, 1, 0)
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms.asnumpy(), result_np)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
@test_utils.run_test_with_On
def test_lin_sapce_dynamic(mode):
    """
    Feature: test dynamic tensor of lin_sapce.
    Description: test dynamic tensor of lin_sapce.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    num_np = 5
    place_holder = ms. Tensor(shape=[None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(lin_space_dyn_shape_func)
    test_cell.set_inputs(place_holder, place_holder)
    start_np = 5
    stop_np = 25
    start = ms.Tensor(start_np, dtype=ms.float32)
    stop = ms.Tensor(stop_np, dtype=ms.float32)
    output = test_cell(start, stop)
    expect = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(output.asnumpy(), expect)
    start_np1 = 15
    stop_np1 = 35
    start1 = ms.Tensor(start_np1, dtype=ms.float32)
    stop1 = ms.Tensor(stop_np1, dtype=ms.float32)
    output1 = test_cell(start1, stop1)
    expect1 = np.linspace(start_np1, stop_np1, num_np)
    assert np.allclose(output1.asnumpy(), expect1)
