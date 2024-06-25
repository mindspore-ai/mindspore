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
def equal_forward_func(x, y):
    return ops.equal(x, y)


@test_utils.run_with_cell
def equal_backward_func(x, y):
    return ops.grad(equal_forward_func, (0,))(x, y)


@test_utils.run_with_cell
def equal_infervalue_func1():
    x = ms.Tensor(np.array([1, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([1, 2, 3]).astype(np.float32))
    return ops.equal(x, y)


@test_utils.run_with_cell
def equal_infervalue_func2():
    x = ms.Tensor(np.array([3, 2, 4]).astype(np.float32))
    y = ms.Tensor(np.array([3]).astype(np.float32))
    return ops.equal(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_equal_forward(mode):
    """
    Feature: Ops.
    Description: test op equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0.0, 1.0, 2.0, -1]), ms.float32)
    y = Tensor(np.array([0.0, 1.0, 2.0, -2]), ms.float32)
    output = equal_forward_func(x, y)
    expect = [True, True, True, False]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_equal_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op equal.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([0.0, 1.0, 2.0, -1]), ms.float32)
    y = Tensor(np.array([0.0, 1.0, 2.0, -2]), ms.float32)
    output = equal_backward_func(x, y)
    expect = [0, 0, 0, 0]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_equal_vmap(mode):
    """
    Feature: test vmap function.
    Description: test equal op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = Tensor(np.array([[[2, 4], [-1, 3]]]), ms.float32)
    y = Tensor(np.array([[[3, 4], [-1, 3]]]), ms.float32)
    nest_vmap = ops.vmap(ops.vmap(equal_forward_func, in_axes=(0, 0)), in_axes=(0, 0))
    output = nest_vmap(x, y)
    expect = [[[False, True], [True, True]]]
    assert np.allclose(output.asnumpy(), expect, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_equal_op_infervalue(context_mode):
    """
    Feature: Ops.
    Description: test op equal infervalue.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    out = equal_infervalue_func1()
    expect_out = np.array([True, True, False]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
    out = equal_infervalue_func2()
    expect_out = np.array([True, False, False]).astype(np.bool)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)
