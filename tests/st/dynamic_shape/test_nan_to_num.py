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
def nan_to_num_forward_func(x, nan, posinf, neginf):
    return ops.NanToNum(nan, posinf, neginf)(x)


@test_utils.run_with_cell
def nan_to_num_backward_func(x, nan, posinf, neginf):
    return ops.grad(nan_to_num_forward_func, (0,))(x, nan, posinf, neginf)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.context.GRAPH_MODE, ms.context.PYNATIVE_MODE])
def test_nan_to_num_forward(mode):
    """
    Feature: Ops.
    Description: test op nan_to_num.
    Expectation: expect correct result.
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), ms.float32)
    out = nan_to_num_forward_func(x, 1.0, 2.0, 3.0)
    expect_out = np.array([1., 2., 3., 3.14])
    assert np.allclose(out.asnumpy(), expect_out)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nan_to_num_backward(mode):
    """
    Feature: Auto grad.
    Description: test auto grad of op nan_to_num.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    x = ms.Tensor(np.array([float('nan'), float('inf'), -float('inf'), 3.14]), ms.float32)
    grads = nan_to_num_backward_func(x, 1.0, 2.0, 3.0)
    expect = np.array([0., 0., 0., 1.]).astype('float32')
    assert np.allclose(grads.asnumpy(), expect)
