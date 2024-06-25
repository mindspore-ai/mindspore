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
from mindspore import ops
import mindspore as ms
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def reshape_forward_func(x, shape):
    return ops.reshape(x, shape)


@test_utils.run_with_cell
def reshape_backward_func(x, shape):
    return ops.grad(reshape_forward_func, (0,))(x, shape)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reshape_op(mode):
    """
    Feature: Ops.
    Description: test op reshape.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.random.rand(1, 3, 12, 12).astype(np.float32)
    x = ms.Tensor(np_array)
    shape = (1, 3, 144)
    out = reshape_forward_func(x, shape)
    expect_out = np_array.reshape(*shape)
    assert (out.asnumpy() == expect_out).all()

    grad = reshape_backward_func(x, shape)
    expect_grad = np.ones((1, 3, 12, 12))
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reshape_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reshape op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = (-1, None)
    np_array = np.random.rand(1, 3, 6, 6, 2, 2).astype(np.float32)
    x = ms.Tensor(np_array)
    shape = (1, 3, 36)
    net_vmap = ops.vmap(ops.vmap(reshape_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = net_vmap(x, shape)
    expect_out = np_array.transpose((5, 4, 0, 1, 2, 3)).reshape((2, 2, 1, 3, 36))
    assert (out.asnumpy() == expect_out).all()
