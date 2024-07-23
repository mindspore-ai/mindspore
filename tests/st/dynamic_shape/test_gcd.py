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
# pylint: disable=unused-variable
import numpy as np
import pytest
from tests.st.utils import test_utils

from mindspore import ops
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def gcd_forward_func(x1, x2):
    return ops.gcd(x1, x2)


@test_utils.run_with_cell
def gcd_backward_func(x1, x2):
    return ops.grad(gcd_forward_func, (0,))(x1, x2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gcd_forward(mode):
    """
    Feature: Ops.
    Description: test op gcd.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    expect_out = np.array([7, 2, 3])
    out = gcd_forward_func(x1, x2)
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gcd_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op gcd when Primitive Gcd's bprop not defined.
    Expectation: expect it raises RuntimeError.
    """
    ms.context.set_context(mode=mode)
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    with pytest.raises(RuntimeError):
        gcd_backward_func(x1, x2)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_gcd_vmap(mode):
    """
    Feature: test vmap function.
    Description: test gcd op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x1 = ms.Tensor(np.array([7, 8, 9]), ms.int32)
    x2 = ms.Tensor(np.array([14, 6, 12]), ms.int32)
    nest_vmap = ops.vmap(gcd_forward_func, in_axes=in_axes, out_axes=0)
    out = nest_vmap(x1, x2)
