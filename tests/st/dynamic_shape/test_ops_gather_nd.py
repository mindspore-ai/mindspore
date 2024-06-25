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
import mindspore as ms
from mindspore import ops, Tensor
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def gather_nd_forward_func(input_x, indices):
    return ops.gather_nd(input_x, indices)


@test_utils.run_with_cell
def gather_nd_backward_func(input_x, indices):
    return ops.grad(gather_nd_forward_func, (0,))(input_x, indices)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_nd_forward(mode):
    """
    Feature: Ops.
    Description: test op gather_nd.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), ms.int32)
    output = gather_nd_forward_func(input_x, indices)
    expect = [-0.1, 0.5]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_nd_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gather_nd.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]), ms.float32)
    indices = Tensor(np.array([[0, 0], [1, 1]]), ms.int32)
    output = gather_nd_backward_func(input_x, indices)
    expect = [[1., 0., 0.], [0., 1., 0.]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_nd_backward_1d(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gather_nd with 1d input.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([1, 2]), ms.int64)
    indices = Tensor(np.array([[1], [0]]), ms.int32)
    output = gather_nd_backward_func(input_x, indices)
    expect = [1, 1]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_gather_nd_vmap(mode):
    """
    Feature: test vmap function.
    Description: test gather_nd op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    input_x = Tensor(np.array([[[[-0.1, 0.3, 3.6], [0.4, 0.5, -3.2]]]]), ms.float32)
    indices = Tensor(np.array([[[[0, 0], [1, 1]]]]), ms.int32)
    nest_vmap = ops.vmap(ops.vmap(gather_nd_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(input_x, indices)
    expect = [[[-0.1, 0.5]]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)
