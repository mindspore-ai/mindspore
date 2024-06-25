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
def reverse_v2_forward_func(x):
    return ops.ReverseV2(axis=(0,))(x)


@test_utils.run_with_cell
def reverse_v2_backward_func(x):
    return ops.grad(reverse_v2_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reverse_v2(mode):
    """
    Feature: Ops.
    Description: test op reverse_v2.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[1, 2, 3, 4], [5, 6, 7, 8]]).astype(np.float32))
    out = reverse_v2_forward_func(x)
    expect_out = np.flip(x.asnumpy(), (0,))
    assert (out.asnumpy() == expect_out).all()

    grad = reverse_v2_backward_func(x)
    expect_grad = np.array([[1, 1, 1, 1],
                            [1, 1, 1, 1]]).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.skip(reason="Probability of failure.")
def test_reverse_v2_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reverse_v2 op vmap.
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
    net_vmap = ops.vmap(ops.vmap(reverse_v2_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = np.array([[[12, 13, 14, 15],
                            [16, 17, 18, 19],
                            [20, 21, 22, 23]],

                           [[0, 1, 2, 3],
                            [4, 5, 6, 7],
                            [8, 9, 10, 11]]]).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_reverse_v2_axis_incorrect():
    """
    Feature: test reverse_v2.
    Description: test reverse_v2 op when axis is incorrect.
    Expectation: raise exception.
    """
    axis = 0.1
    with pytest.raises(TypeError) as err:
        ops.ReverseV2(axis)
    assert "For 'ReverseV2', the type of 'axis' should be one of '[list of int, tuple of int]'" in str(err.value)
