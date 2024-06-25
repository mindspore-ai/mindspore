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
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def next_after_forward_func(x, other):
    return ops.nextafter(x, other)


@test_utils.run_with_cell
def next_after_backward_func(x, other):
    return ops.grad(next_after_forward_func, 0)(x, other)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_next_after_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op next_after forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([0.0]).astype(data_type))
    other = ms.Tensor(np.array([0.1]).astype(data_type))
    out = next_after_forward_func(x, other)
    expect_out = np.array([1.e-45]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.skip(reason="dynamic shape not support now")
@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_next_after_op_forward_cpu(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op next_after forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([0.0]).astype(data_type))
    other = ms.Tensor(np.array([0.1]).astype(data_type))
    out = next_after_forward_func(x, other)
    expect_out = np.array([0.]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_next_after_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op next_after.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([0.0]).astype(data_type))
    other = ms.Tensor(np.array([0.1]).astype(data_type))
    grads = next_after_backward_func(x, other)
    expect_out = np.array([1.]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nextafter_dtype_diff(context_mode):
    """
    Feature: type valid.
    Description: test different input type op next_after.
    Expectation: expect raise exception.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.random.randn(), ms.float32)
    other = ms.Tensor(np.random.randint(0, 100000, ()), ms.float64)
    with pytest.raises((RuntimeError, TypeError)):
        next_after_forward_func(x, other)
