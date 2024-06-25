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
def reduce_prod_forward_func(x):
    return ops.ReduceProd(keep_dims=False)(x, 1)


@test_utils.run_with_cell
def reduce_prod_backward_func(x):
    return ops.grad(reduce_prod_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reduce_prod(mode):
    """
    Feature: Ops.
    Description: test op reduce prod.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[0.392192, 1.484581, 1.111067],
                            [1.614102, 0.676057, 3.279313]]).astype(np.float32))
    out = reduce_prod_forward_func(x)
    expect_out = np.array([[0.646909, 3.578468]]).astype(np.float32)
    assert np.allclose(out.asnumpy(), expect_out, rtol=1e-4, atol=1e-4)

    grad = reduce_prod_backward_func(x)
    expect_grad = np.array([[1.649469, 0.435752, 0.582241],
                            [2.217002, 5.293146, 1.091225]]).astype(np.float32)
    assert np.allclose(grad.asnumpy(), expect_grad, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reduce_prod_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reduce_prod op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.random.uniform(low=-1, high=1, size=(4, 3, 2, 2)).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(reduce_prod_forward_func, in_axes=in_axes, out_axes=-1), in_axes=in_axes, out_axes=-1)
    out = nest_vmap(x)
    expect_out = reduce_prod_forward_func(x)
    assert np.allclose(out.asnumpy(), expect_out.asnumpy(), rtol=1e-4, atol=1e-4)
