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
def fastgelugrad_forward_func(dy, x):
    return ops.auto_generate.FastGeLUGrad()(dy, x)


@test_utils.run_with_cell
def fastgelugrad_vmap_func(dy, x):
    return ops.vmap(fastgelugrad_forward_func, in_axes=0, out_axes=0)(dy, x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelugrad_op_forward(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op fast_gelu_grad forward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    dy = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    out = fastgelugrad_forward_func(dy, x)
    expect_out = np.array([1.06778, 2.14763, 3.073645]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelugrad_op_forward_ascend(context_mode, data_type):
    """
    Feature: Ops.
    Description: test op fast_gelu_grad forward for ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    dy = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    out = fastgelugrad_forward_func(dy, x)
    expect_out = np.array([1.069909, 2.145001, 3.068479]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-2)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelugrad_op_vmap(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test fastgelugrad op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    dy = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    out = fastgelugrad_vmap_func(dy, x)
    expect_out = np.array([1.06778, 2.14763, 3.073645]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE])
@pytest.mark.parametrize("data_type", [np.float32])
def test_fastgelugrad_op_vmap_ascend(context_mode, data_type):
    """
    Feature: test vmap function.
    Description: test fastgelugrad op vmap for ascend.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    dy = ms.Tensor(np.array([1, 2, 3]).astype(data_type))
    out = fastgelugrad_vmap_func(dy, x)
    expect_out = np.array([1.069909, 2.145001, 3.068479]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-2)
