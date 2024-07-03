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
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops import erfinv

import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return Tensor([0, 0.47693613, -1.1630869], dtype=mstype.float32)


def generate_expect_backward_output(x):
    return Tensor([0.88622695, 1.1125847, 3.428041], dtype=mstype.float32)

@test_utils.run_with_cell
def erfinv_forward_func(x):
    return erfinv(x)

@test_utils.run_with_cell
def erfinv_backward_func(x):
    return ops.grad(erfinv_forward_func, (0))(x)

@test_utils.run_with_cell
def erfinv_vmap_func(x):
    return ops.vmap(erfinv_forward_func)(x)

@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_erfinv_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function erfinv forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = Tensor([0, 0.5, -0.9], ms.float32)
    output = erfinv(x)
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-3)

    output = erfinv_backward_func(ms.Tensor(x))
    expect = generate_expect_backward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_erfinv_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function erfinv vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = Tensor([0, 0.5, -0.9], ms.float32)
    output = erfinv_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect.asnumpy(), rtol=1e-3)
