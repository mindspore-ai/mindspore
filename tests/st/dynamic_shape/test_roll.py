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
def roll_forward_func(x):
    return ops.Roll(shift=1, axis=0)(x)


@test_utils.run_with_cell
def roll_backward_func(x):
    return ops.grad(roll_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_roll(mode):
    """
    Feature: Ops.
    Description: test op roll.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([0, 1, 2, 3, 4]).astype(np.float32))
    out = roll_forward_func(x)
    expect_out = np.array([4, 0, 1, 2, 3]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()

    grad = roll_backward_func(x)
    expect_grad = np.array([1, 1, 1, 1, 1]).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_roll_vmap(mode):
    """
    Feature: test vmap function.
    Description: test roll op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = -1
    x = ms.Tensor(np.array([[[0, 1, 2, 3],
                             [4, 5, 6, 7],
                             [8, 9, 10, 11]],

                            [[12, 13, 14, 15],
                             [16, 17, 18, 19],
                             [20, 21, 22, 23]]]).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(roll_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = np.array([[[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]],

                           [[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11]]]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()
