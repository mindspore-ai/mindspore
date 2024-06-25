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
def oneslike_forward_func(x):
    return ops.OnesLike()(x)


@test_utils.run_with_cell
def oneslike_backward_func(x):
    return ops.grad(oneslike_forward_func, 0)(x)


@test_utils.run_with_cell
def oneslike_vmap_func(x):
    return ops.vmap(oneslike_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_oneslike_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op oneslike forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.random.uniform(-2, 2, (2, 3)).astype(data_type))
    out = oneslike_forward_func(x)
    expect_out = np.array([[1., 1., 1.], [1., 1., 1.]]).astype(np.float32)
    np.testing.assert_array_equal(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_oneslike_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op oneslike.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.random.uniform(-2, 2, (2, 3)).astype(data_type))
    grads = oneslike_backward_func(x)
    expect_out = np.array([[0., 0., 0.], [0., 0., 0.]]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
@test_utils.run_test_with_On
def test_oneslike_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test oneslike op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.random.uniform(-2, 2, (2, 3)).astype(data_type))
    out = oneslike_vmap_func(x)
    expect_out = np.array([[1., 1., 1.], [1., 1., 1.]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
