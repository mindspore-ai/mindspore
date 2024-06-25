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
def floor_mod_forward_func(x, y):
    return ops.floor_mod(x, y)


@test_utils.run_with_cell
def floor_mod_backward_func(x, y):
    return ops.grad(floor_mod_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_floor_mod_forward(mode):
    """
    Feature: Ops.
    Description: test op floor_mod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(2.0, ms.float32)
    y = Tensor(2.0, ms.float32)
    output = floor_mod_forward_func(x, y)
    expect = 0.0
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_floor_mod_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op floor_mod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([2, 4, -1]), ms.int32)
    y = Tensor(np.array([3, 3, 3]), ms.int32)
    output = floor_mod_backward_func(x, y)
    expect = [1, 1, 1]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_floor_mod_vmap(mode):
    """
    Feature: test vmap function.
    Description: test floor_mod op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[2, 4], [-1, 3]]]))
    y = Tensor(np.array([[[3, 3], [3, 3]]]))
    nest_vmap = ops.vmap(ops.vmap(floor_mod_forward_func, in_axes=(0, 0)), in_axes=(0, 0))
    output = nest_vmap(x, y)
    expect = [[[2, 1], [2, 0]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
