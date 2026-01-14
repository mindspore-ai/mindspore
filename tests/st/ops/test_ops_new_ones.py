# Copyright 2024 Huawei Technologies Co., Ltd
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

import mindspore as ms
from mindspore import dtype as mstype
from mindspore.ops.function.array_func import new_ones

import tests.st.utils.test_utils as test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, size, dtype):
    if dtype is None:
        dtype = x.dtype
    return np.ones(size, mstype._dtype_to_nptype(dtype))  # pylint:disable=protected-access


def generate_expect_backward_output():
    return 0


@test_utils.run_with_cell
def new_ones_forward_func(x, size, dtype=None):
    return new_ones(x, size, dtype=dtype)


@test_utils.run_with_cell
def new_ones_backward_func(x, size, dtype=None):
    return ms.grad(new_ones_forward_func, (0,))(x, size, dtype=dtype)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('context_mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('size', [0, 1, (2, 3)])
@pytest.mark.parametrize('dtype', [None, mstype.int32])
@test_utils.run_test_with_On
def test_ops_new_ones_normal(context_mode, size, dtype):
    """
    Feature: pyboost function.
    Description: test function new_ones forward and backward.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = new_ones_forward_func(ms.Tensor(x), size, dtype)
    expect = generate_expect_forward_output(ms.Tensor(x), size, dtype)
    assert np.allclose(output.asnumpy(), expect, rtol=1e-5)

    output_grad = new_ones_backward_func(ms.Tensor(x), size, dtype)
    expect_grad = generate_expect_backward_output()
    assert np.allclose(output_grad.asnumpy(), expect_grad, rtol=1e-5)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=3072.00)
def test_new_ones_dyn_shape():
    """
    Feature: Test dynamic shape.
    Description: test function new_ones dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5), np.float32)
    size1 = (1, 2)
    ms_data2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    size2 = (2, 3, 4)
    TEST_OP(new_ones_forward_func, [[ms.Tensor(ms_data1), size1], [ms.Tensor(ms_data2), size2]])
