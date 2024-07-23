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
def erfc_forward_func(x):
    return ops.erfc(x)


@test_utils.run_with_cell
def erfc_backward_func(x):
    return ops.grad(erfc_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfc_forward(mode):
    """
    Feature: Ops.
    Description: test op erfc.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([-2, -1, 0, 1, 2]), ms.float32)
    output = erfc_forward_func(x)
    expect = [1.9953222, 1.8427008, 1., 0.1572992, 0.00467774]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfc_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op erfc.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([-1, 0, 1]), ms.float32)
    output = erfc_backward_func(x)
    expect = np.array([-0.41510752, -1.1283791, -0.41510752])
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erfc_vmap(mode):
    """
    Feature: test vmap function.
    Description: test erfc op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[-2., -1.], [0., 1.]]]))
    nest_vmap = ops.vmap(ops.vmap(erfc_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(x)
    expect = [[[1.9953222, 1.8427008], [1., 0.1572992]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
