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
def reduce_all_forward_func(x):
    return ops.ReduceAll(True)(x, 0)


@test_utils.run_with_cell
def reduce_all_backward_func(x):
    return ops.grad(reduce_all_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reduce_all(mode):
    """
    Feature: Ops.
    Description: test op reduce all.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([[True, True, False],
                            [False, True, False]]))
    out = reduce_all_forward_func(x)
    expect_out = np.array([[False, True, False]])
    assert (out.asnumpy() == expect_out).all()

    grad = reduce_all_backward_func(x)
    expect_grad = np.array([[0, 0, 0], [0, 0, 0]])
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_reduce_all_vmap(mode):
    """
    Feature: test vmap function.
    Description: test reduce_all op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    in_axes = -1
    x = ms.Tensor(np.random.randint(low=0, high=2, size=(4, 3, 2, 2)).astype(np.bool))
    nest_vmap = ops.vmap(ops.vmap(reduce_all_forward_func, in_axes=in_axes, out_axes=-1), in_axes=in_axes, out_axes=-1)
    out = nest_vmap(x)
    expect_out = reduce_all_forward_func(x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
