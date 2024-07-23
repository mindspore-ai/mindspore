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
def erfinv_forward_func(x):
    return ops.erfinv(x)


@test_utils.run_with_cell
def erfinv_backward_func(x):
    return ops.grad(erfinv_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfinv_forward(mode):
    """
    Feature: Ops.
    Description: test op erfinv.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0, 0.5, -0.9]), ms.float32)
    output = erfinv_forward_func(x)
    expect = [0., 0.47695306, -1.1630805]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfinv_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op erfinv.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0, 0.5, -0.8]), ms.float32)
    output = erfinv_backward_func(x)
    expect = np.array([0.88622695, 1.112585, 2.0145686])
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfinv_vmap(mode):
    """
    Feature: test vmap function.
    Description: test erfinv op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[0.5, 0], [0.5, -0.9]]]), ms.float32)
    nest_vmap = ops.vmap(ops.vmap(erfinv_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(x)
    expect = [[[0.47695306, 0.], [0.47695306, -1.1630805]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
