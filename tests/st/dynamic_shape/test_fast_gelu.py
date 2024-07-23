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
def fastgelu_forward_func(x):
    return ops.fast_gelu(x)


@test_utils.run_with_cell
def fastgelu_backward_func(x):
    return ops.grad(fastgelu_forward_func, 0)(x)


@test_utils.run_with_cell
def fastgelu_vmap_func(x):
    return ops.vmap(fastgelu_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op fastgelu forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    out = fastgelu_forward_func(x)
    expect_out = np.array([0.84579575, 1.9356586, 2.9819288]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_forward_ascend(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op fastgelu forward for ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    out = fastgelu_forward_func(x)
    expect_out = np.array([0.845703, 1.9375, 2.982422]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_backward(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op fastgelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    grads = fastgelu_backward_func(x)
    expect_out = np.array([1.0677795, 1.0738152, 1.0245484]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_backward_ascend(context_mode, data_type):
    """
    Feature: Auto grad.
    Description: test auto grad of op fastgelu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    grads = fastgelu_backward_func(x)
    expect_out = np.array([1.069909, 1.072501, 1.022826]).astype(np.float32)
    np.testing.assert_allclose(grads.asnumpy(), expect_out, rtol=1e-2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test fastgelu op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    out = fastgelu_vmap_func(x)
    expect_out = np.array([0.84579575, 1.9356586, 2.9819288]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelu_op_vmap_ascend(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test fastgelu op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1., 2., 3.]).astype(data_type))
    out = fastgelu_vmap_func(x)
    expect_out = np.array([0.845703, 1.9375, 2.982422]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-2)
