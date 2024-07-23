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

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def cosh_forward_func(x):
    return ops.cosh(x)


@test_utils.run_with_cell
def cosh_backward_func(x):
    return ops.grad(cosh_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_cosh_forward(mode):
    """
    Feature: Ops.
    Description: test op cosh.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    x = Tensor(np_array)
    out = cosh_forward_func(x)
    expect = np.cosh(np_array)
    assert np.allclose(out.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_cosh_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op cosh.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([1, 0.5, -0.5, 0.3]).astype('float32')
    x = Tensor(np_array)
    grads = cosh_backward_func(x)
    expect = np.array([1.17520118e+00, 5.21095276e-01, -5.21095276e-01, 3.04520309e-01]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_cosh_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([[0.5, 0.4, -0.3, -0.2]]).astype('float32')
    x = Tensor(np_array)
    nest_vmap = ops.vmap(ops.vmap(cosh_forward_func, in_axes=0), in_axes=0)
    out = nest_vmap(x)
    expect = np.array([[1.127626, 1.0810723, 1.0453385, 1.0200667]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect, rtol=1e-4, atol=1e-4)
