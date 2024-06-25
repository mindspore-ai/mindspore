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
import mindspore as ms
from mindspore import ops
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def rsqrt_grad_func(dy, x):
    return ops.auto_generate.RsqrtGrad()(dy, x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rsqrt_grad(mode):
    """
    Feature: Ops.
    Description: test op rsqrt grad.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy = ms.Tensor(np.array([[[[-1, 1, 10],
                               [5.9, 6.1, 6],
                               [10, 1, -1]]]]).astype(np.float32))
    x = ms.Tensor(np.array([[[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]]]]).astype(np.float32))
    out = rsqrt_grad_func(dy, x)
    expect_out = np.array([[[[0.5, -0.5, -500],
                             [-205.37901, -226.98099, -216],
                             [-1500, -1.5, 1.5]]]]).astype(np.float32)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rsqrt_grad_dyn_rank(mode):
    """
    Feature: Ops.
    Description: test op rsqrt_grad dynamic rank.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    dy_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(rsqrt_grad_func)
    test_cell.set_inputs(dy_dyn, x_dyn)
    dy = ms.Tensor(np.array([[[[-1, 1, 10],
                               [5.9, 6.1, 6],
                               [10, 1, -1]]]]).astype(np.float32))
    x = ms.Tensor(np.array([[[[1, 1, 1],
                              [2, 2, 2],
                              [3, 3, 3]]]]).astype(np.float32))
    expect_out = np.array([[[[0.5, -0.5, -500],
                             [-205.37901, -226.98099, -216],
                             [-1500, -1.5, 1.5]]]]).astype(np.float32)
    out = test_cell(dy, x)
    np.testing.assert_allclose(out.asnumpy(), expect_out, rtol=1e-3)
    dy1 = ms.Tensor(np.array([-1, 1, 10, 5.9, 6.1, 6, 10, 1, -1]).astype(np.float32))
    x1 = ms.Tensor(np.array([1, 1, 1, 2, 2, 2, 3, 3, 3]).astype(np.float32))
    expect_out1 = np.array([0.5, -0.5, -500, -205.37901, -226.98099, -216, -1500, -1.5, 1.5]).astype(np.float32)
    out1 = test_cell(dy1, x1)
    np.testing.assert_allclose(out1.asnumpy(), expect_out1, rtol=1e-3)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_rsqrt_grad_vmap(mode):
    """
    Feature: test vmap function.
    Description: test rsqrt op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = (-1, -1)
    dy = ms.Tensor(np.random.rand(4, 3, 2).astype(np.float32))
    x = ms.Tensor(np.random.rand(4, 3, 2).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(rsqrt_grad_func, in_axes=axes, out_axes=-1), in_axes=axes, out_axes=-1)
    out = net_vmap(dy, x)
    expect_out = rsqrt_grad_func(dy, x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
