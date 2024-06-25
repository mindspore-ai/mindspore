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
def transpose_forward_func(x):
    return ops.transpose(x, (1, 2, 0))


@test_utils.run_with_cell
def transpose_backward_func(x):
    return ops.grad(transpose_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_transpose_forward(mode):
    """
    Feature: Ops.
    Description: test op relu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(24).reshape(4, 3, 2).astype(np.float32))
    out = transpose_forward_func(x)
    expect = np.array([[[[0, 6, 12, 18],
                         [1, 7, 13, 19]],
                        [[2, 8, 14, 20],
                         [3, 9, 15, 21]],
                        [[4, 10, 16, 22],
                         [5, 11, 17, 23]]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_transpose_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op relu.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.arange(24).reshape(4, 3, 2).astype(np.float32))
    grad = transpose_backward_func(x)
    expect_grad = np.ones((4, 3, 2)).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE])
@test_utils.run_test_with_On
def test_transpose_vmap(mode):
    """
    Feature: test vmap function.
    Description: test avgpool op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.arange(24).reshape(2, 3, 1, 2, 2).astype(np.float32))
    nest_vmap = ops.vmap(ops.vmap(transpose_forward_func, in_axes=in_axes, out_axes=0), in_axes=in_axes, out_axes=0)
    out = nest_vmap(x)
    expect = np.array([[[[[0, 12]],
                         [[4, 16]],
                         [[8, 20]]],
                        [[[2, 14]],
                         [[6, 18]],
                         [[10, 22]]]],
                       [[[[1, 13]],
                         [[5, 17]],
                         [[9, 21]]],
                        [[[3, 15]],
                         [[7, 19]],
                         [[11, 23]]]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()
