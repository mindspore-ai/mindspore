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
def right_shift_forward_func(x, y):
    return ops.RightShift()(x, y)


@test_utils.run_with_cell
def right_shift_backward_func(x, y):
    return ops.grad(right_shift_forward_func, (0,))(x, y)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_right_shift_forward(mode):
    """
    Feature: Ops.
    Description: test op right_shift.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3], [3, 2, 1]]).astype(np.uint32))
    y = ms.Tensor(np.array([1, 1, 1]).astype(np.uint32))
    out = right_shift_forward_func(x, y)
    expect_out = np.array([[0, 1, 1], [1, 1, 0]]).astype(np.uint32)
    assert (out.asnumpy() == expect_out).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_right_shift_vmap(mode):
    """
    Feature: test vmap function.
    Description: test right_shift op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = (-1, -1)
    x = ms.Tensor(np.random.randint(low=0, high=2, size=(4, 3, 2)).astype(np.uint32))
    y = ms.Tensor(np.random.randint(low=1, high=10, size=(4, 3, 2)).astype(np.uint32))
    net_vmap = ops.vmap(ops.vmap(right_shift_forward_func, in_axes=axes, out_axes=-1), in_axes=axes, out_axes=-1)
    out = net_vmap(x, y)
    expect_out = right_shift_forward_func(x, y)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
