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

"""Test ZerosLike"""

import numpy as np
import pytest
from mindspore import ops
import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def zeros_like_forward_func(x):
    return ops.ZerosLike()(x)


@test_utils.run_with_cell
def zeros_like_backward_func(x):
    return ops.grad(zeros_like_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_zeros_like_forward(mode):
    """
    Feature: Ops.
    Description: test op zeros_like.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    shape = (4, 3, 2)
    x = ms.Tensor(np.ones(shape).astype(np.float32))
    out = zeros_like_forward_func(x)
    expect = np.zeros(shape).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_zeros_like_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op zeros_like.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    shape = (4, 3, 2)
    x = ms.Tensor(np.ones(shape).astype(np.float32))
    grad = zeros_like_backward_func(x)
    expect_grad = np.zeros(shape).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_zeros_like_vmap(mode):
    """
    Feature: test vmap function.
    Description: test zeros_like op vmap.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.ones((4, 3, 2, 2, 2)).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(zeros_like_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    expect = np.zeros((2, 2, 4, 3, 2)).astype(np.float32)
    assert (out.asnumpy() == expect).all()
