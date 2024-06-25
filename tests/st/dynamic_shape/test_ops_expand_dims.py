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
def expand_dims_forward_func(x, axis):
    return ops.expand_dims(x, axis)


@test_utils.run_with_cell
def expand_dims_backward_func(x, axis):
    return ops.grad(expand_dims_forward_func, (0,))(x, axis)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expand_dims_forward(mode):
    """
    Feature: Ops.
    Description: test op expand_dims.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    axis = 0
    output = expand_dims_forward_func(x, axis)
    expect = [[[2., 2.], [2., 2.]]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expand_dims_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op expand_dims.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[2, 2], [2, 2]]), ms.float32)
    axis = 0
    output = expand_dims_backward_func(x, axis)
    expect = [[1., 1.], [1., 1.]]
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expand_dims_vmap(mode):
    """
    Feature: test vmap function.
    Description: test expand_dims op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[[2, 2], [2, 2]]]]), ms.float32)
    axis = 0
    nest_vmap = ops.vmap(ops.vmap(expand_dims_forward_func, in_axes=(0, None)), in_axes=(0, None))
    output = nest_vmap(x, axis)
    expect = [[[[[2., 2.], [2., 2.]]]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
