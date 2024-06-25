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
def expm1_forward_func(x):
    return ops.expm1(x)


@test_utils.run_with_cell
def expm1_backward_func(x):
    return ops.grad(expm1_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expm1_forward(mode):
    """
    Feature: Ops.
    Description: test op expm1.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0.0, 1.0, 2.0, -1]), ms.float32)
    output = expm1_forward_func(x)
    expect = [0., 1.718282, 6.389056, -0.63212055]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expm1_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op expm1.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([1, -1]), ms.float32)
    output = expm1_backward_func(x)
    expect = [2.7182817, 0.36787948]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_expm1_vmap(mode):
    """
    Feature: test vmap function.
    Description: test expm1 op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[0.0, 1.0], [2.0, -1]]]))
    nest_vmap = ops.vmap(ops.vmap(expm1_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(x)
    expect = [[[0., 1.718282], [6.389056, -0.63212055]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
