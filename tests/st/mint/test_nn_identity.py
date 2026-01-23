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
from mindspore import  ops, mint
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_random_input(shape, dtype):
    return np.random.randint(1, 10, size=shape).astype(dtype)


@test_utils.run_with_cell
def identity_forward_func(input_x):
    net = mint.nn.Identity()
    return net(input_x)


def generate_expect_forward_output(x):
    return x


@test_utils.run_with_cell
def identity_backward_func(input_x):
    return ops.grad(identity_forward_func, (0))(input_x)


def identity_func(x):
    net = mint.nn.Identity()
    return net(x)


@test_utils.run_with_cell
def identity_vmap_func(x):
    return ops.vmap(identity_forward_func, in_axes=0, out_axes=0)(x)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_identity_forward_func_normal(context_mode):
    """
    Feature: pyboost function.
    Description: test function identity forward and backward.
    Expectation: expect correct result.
    """

    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = identity_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

    output = identity_backward_func(ms.Tensor(x))
    assert np.all(output.asnumpy() == 1)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=0.01)
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_nn_identity_forward_func_vmap(context_mode):
    """
    Feature: pyboost function.
    Description: test function identity vmap feature.
    Expectation: expect correct result.
    """
    ms.context.set_context(mode=context_mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    output = identity_vmap_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential', mem_peak=1024.0)
def test_nn_identity_func_dynamic():
    """
    Feature: pyboost function.
    Description: test function identity with dynamic shape and dynamic rank.
    Expectation: expect correct result.
    """
    input1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2 = generate_random_input((3, 3, 4), np.float32)
    TEST_OP(identity_func, [[ms.Tensor(input1)], [ms.Tensor(input2)]])
