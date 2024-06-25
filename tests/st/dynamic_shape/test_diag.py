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

from mindspore import ops, context
import mindspore as ms
from tests.mark_utils import arg_mark


@test_utils.run_with_cell
def diag_forward_func(input_x):
    return ops.diag(input_x)


@test_utils.run_with_cell
def diag_backward_func(input_x):
    return ops.grad(diag_forward_func, (0,))(input_x)


@test_utils.run_with_cell
def diag_vmap_func(x):
    return ops.vmap(diag_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_diag_forward(mode):
    """
    Feature: Ops.
    Description: test op diag.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    input_x = ms.Tensor(np.array([1, 2, 5]), ms.float16)
    out = diag_forward_func(input_x)
    expect = np.array([[1, 0, 0],
                       [0, 2, 0],
                       [0, 0, 5]]).astype(np.float16)
    assert (out.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu', 'cpu_linux'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_diag_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op diag.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    input_x = ms.Tensor([[1.3, 2.5, -2.1], [-4.7, 2.5, 1.0]], ms.float32)
    out = diag_backward_func(input_x)
    expect = np.array([[1, 1, 1],
                       [1, 1, 1]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()


@arg_mark(plat_marks=['platform_gpu'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_diag_vmap(mode):
    """
    Feature: test vmap function.
    Description: test diag op vmap.
    Expectation: expect correct result.
    """
    context.set_context(mode=mode)
    input_x = ms.Tensor([[3, 1, 4], [1, 5, 9]], ms.float32)
    out = diag_vmap_func(input_x)
    expect = np.array([[[3, 0, 0], [0, 1, 0], [0, 0, 4]], [[1, 0, 0], [0, 5, 0], [0, 0, 9]]]).astype(np.float32)
    assert (out.asnumpy() == expect).all()
