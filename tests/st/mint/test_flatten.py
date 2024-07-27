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

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
import mindspore.mint as mint
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def flatten_func(x, start_dim=0, end_dim=-1):
    return mint.flatten(x, start_dim, end_dim)


@test_utils.run_with_cell
def flatten_forward_func(x, start_dim=0, end_dim=-1):
    return flatten_func(x, start_dim, end_dim)


def flatten_bwd_func(x, start_dim=0, end_dim=-1):
    return ops.grad(flatten_func, (0,))(x, start_dim, end_dim)


@test_utils.run_with_cell
def flatten_backward_func(x, start_dim=0, end_dim=-1):
    return flatten_bwd_func(x, start_dim, end_dim)


@pytest.mark.level3
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_flatten_forward(mode):
    """
    Feature: Ops.
    Description: test op flatten.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    output = flatten_forward_func(ms.Tensor(x), 1, 2)
    expect = x.reshape((2, 12, 5))
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    output2 = flatten_forward_func(ms.Tensor(x), 0, 2)
    expect2 = x.reshape((24, 5))
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@pytest.mark.level3
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_flatten_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4)
    x = generate_random_input(test_shape, np.float32)
    output = flatten_forward_func(ms.Tensor(x))
    expect = x.reshape(24)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)

    output2 = flatten_forward_func(ms.Tensor(x), 0, 1)
    expect2 = x.reshape((6, 4))
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=5e-3, atol=5e-3)


@pytest.mark.level3
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_flatten_backward(mode):
    """
    Feature: Ops.
    Description: test op flatten.
    Expectation: expect correct result.
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    x = generate_random_input(test_shape, np.float32)
    output = flatten_backward_func(ms.Tensor(x), 1, 3)
    expect = np.ones(test_shape).astype(np.float32)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    output2 = flatten_backward_func(ms.Tensor(x), 0, 2)
    expect2 = np.ones(test_shape).astype(np.float32)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)


@pytest.mark.level3
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_flatten_dynamic_shape():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    ms_data1 = generate_random_input((2, 3, 4, 5, 6), np.float32)
    ms_data2 = generate_random_input((3, 4, 5, 6), np.float32)
    TEST_OP(flatten_forward_func, [[ms.Tensor(ms_data1), 2, 3],
                                   [ms.Tensor(ms_data2), 0, 1]], '', disable_yaml_check=True,
            disable_mode=['GRAPH_MODE'])
