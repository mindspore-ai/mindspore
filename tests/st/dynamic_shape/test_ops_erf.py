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
def erf_forward_func(x):
    return ops.erf(x)


@test_utils.run_with_cell
def erf_backward_func(x):
    return ops.grad(erf_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erf_forward(mode):
    """
    Feature: Ops.
    Description: test op erf.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([-1, 0, 1, 2, 3]), ms.float32)
    output = erf_forward_func(x)
    expect = np.array([-0.8427168, 0., 0.8427168, 0.99530876, 0.99997765], dtype=np.float32)
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erf_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op erf.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([-1, 0, 1, 2, 3]), ms.float32)
    output = erf_backward_func(x)
    expect = np.array([4.1510752e-01, 1.1283791e+00, 4.1510752e-01, 2.0666985e-02, 1.3925304e-04])
    assert np.allclose(output.asnumpy(), expect, rtol=0.001)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_erf_vmap(mode):
    """
    Feature: test vmap function.
    Description: test erf op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[-1., 0.], [1., 2.]]]))
    nest_vmap = ops.vmap(ops.vmap(erf_forward_func, in_axes=0), in_axes=0)
    output = nest_vmap(x)
    expect = [[[-0.8427168, 0.], [0.8427168, 0.99530876]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)
