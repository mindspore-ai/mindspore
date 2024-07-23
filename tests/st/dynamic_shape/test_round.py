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

ms.context.set_context(ascend_config={"precision_mode": "force_fp32"})

@test_utils.run_with_cell
def round_forward_func(x):
    return ops.round(x)


@test_utils.run_with_cell
def round_backward_func(x):
    return ops.grad(round_forward_func, (0,))(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_round(mode):
    """
    Feature: Ops.
    Description: test op round.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    np_array = np.array([0.8, 1.5, 2.3, 2.5, -4.5]).astype(np.float32)
    x = ms.Tensor(np_array)
    out = round_forward_func(x)
    expect_out = np.round(np_array).astype(np.float32)
    assert (out.asnumpy() == expect_out).all()

    grad = round_backward_func(x)
    expect_grad = np.zeros(shape=(5,)).astype(np.float32)
    assert (grad.asnumpy() == expect_grad).all()


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@test_utils.run_test_with_On
def test_round_vmap(mode):
    """
    Feature: test vmap function.
    Description: test round op vmap.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=mode)
    axes = -1
    x = ms.Tensor(np.random.uniform(low=-255, high=255, size=(4, 3, 2)).astype(np.float32))
    net_vmap = ops.vmap(ops.vmap(round_forward_func, in_axes=axes, out_axes=axes), in_axes=axes, out_axes=axes)
    out = net_vmap(x)
    expect_out = round_forward_func(x)
    assert (out.asnumpy() == expect_out.asnumpy()).all()
