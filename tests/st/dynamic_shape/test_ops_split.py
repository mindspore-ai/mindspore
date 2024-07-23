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
def split_forward_func(x, axis=0, output_num=2):
    return ops.Split(axis, output_num)(x)


@test_utils.run_with_cell
def split_backward_func(x, axis, output_num):
    return ops.grad(split_forward_func, (0,))(x, axis, output_num)


def split_dyn_shape_func(x, axis=0, output_num=2):
    return ops.Split(axis, output_num)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_forward(mode):
    """
    Feature: Ops.
    Description: test op split.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x = np.array(np.arange(20).reshape((10, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    out = split_forward_func(x, 0, 2)
    expect = [np.array(np.arange(10).reshape((5, 2)), dtype=np.float32),
              np.array(np.arange(10, 20).reshape((5, 2)), dtype=np.float32)]
    for res, exp in zip(out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op split.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x = np.array(np.arange(20).reshape((5, 2, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    grads = split_backward_func(x, 2, 2)
    expect = np.ones((5, 2, 2))
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_vmap(mode):
    """
    Feature: test vmap function.
    Description: test split op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_x = np.array(np.arange(6).reshape((3, 2)), dtype=np.float32)
    x = ms.Tensor(np_x, dtype=ms.float32)
    in_axes = (0)
    nest_vmap = ops.vmap(split_forward_func, in_axes=in_axes, out_axes=0)
    vmap_out = nest_vmap(x)
    expect = (np.array([[0], [2], [4]]), np.array([[1], [3], [5]]))
    for res, exp in zip(vmap_out, expect):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_dynamic(mode):
    """
    Feature: test dynamic tensor of split.
    Description: test dynamic tensor of split.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(split_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x1 = np.arange(2 * 2).reshape(2, 2)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = (np.array([[0, 1]]), np.array([[2, 3]]))
    for res, exp in zip(output1, expect1):
        assert np.allclose(res.asnumpy(), exp)
    np_x2 = np.arange(2 * 3).reshape(2, 3)
    x2 = ms.Tensor(np_x2, ms.float32)
    output2 = test_cell(x2)
    expect2 = (np.array([[0, 1, 2]]), np.array([[3, 4, 5]]))
    for res, exp in zip(output2, expect2):
        assert np.allclose(res.asnumpy(), exp)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.PYNATIVE_MODE, ms.context.GRAPH_MODE])
def test_split_dynamic_rank(mode):
    """
    Feature: test dynamic rank tensor of split.
    Description: test dynamic rank tensor of split.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(split_dyn_shape_func)
    test_cell.set_inputs(x_dyn)
    np_x1 = np.arange(2 * 2).reshape(2, 2)
    x1 = ms.Tensor(np_x1, ms.float32)
    output1 = test_cell(x1)
    expect1 = (np.array([[0, 1]]), np.array([[2, 3]]))
    for res, exp in zip(output1, expect1):
        assert np.allclose(res.asnumpy(), exp)
    np_x2 = np.arange(2 * 3).reshape(2, 3)
    x2 = ms.Tensor(np_x2, ms.float32)
    output2 = test_cell(x2)
    expect2 = (np.array([[0, 1, 2]]), np.array([[3, 4, 5]]))
    for res, exp in zip(output2, expect2):
        assert np.allclose(res.asnumpy(), exp)
