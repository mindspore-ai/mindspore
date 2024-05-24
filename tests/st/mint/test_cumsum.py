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

import os
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.mint import cumsum
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils

def generate_random_input(shape, dim):
    x = np.random.randn(*shape)
    return x, np.cumsum(x, dim)


def cumsum_func(x, dim):
    return cumsum(x, dim)


@test_utils.run_with_cell
def cumsum_forward_func(x, dim):
    return cumsum_func(x, dim)


def cumsum_bwd_func(x, dim):
    return ops.grad(cumsum_func, (0,))(x, dim)


@test_utils.run_with_cell
def cumsum_backward_func(x, dim):
    return cumsum_bwd_func(x, dim)


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_forward(mode):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    dim1 = 2
    x, expect = generate_random_input(test_shape, dim1)
    output = cumsum_forward_func(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    dim2 = 0
    x, expect2 = generate_random_input(test_shape, dim2)
    output2 = cumsum_forward_func(ms.Tensor(x), dim2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)
    del os.environ['GRAPH_OP_RUN']


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4)
    dim1 = 1
    x, expect = generate_random_input(test_shape, dim1)
    output = cumsum_forward_func(ms.Tensor(x), dim1)
    np.testing.assert_allclose(output.float().asnumpy(), expect, rtol=5e-3, atol=5e-3)

    dim2 = 2
    x, expect2 = generate_random_input(test_shape, dim2)
    output2 = cumsum_forward_func(ms.Tensor(x), dim2)
    np.testing.assert_allclose(output2.float().asnumpy(), expect2, rtol=5e-3, atol=5e-3)
    del os.environ['GRAPH_OP_RUN']


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_backward(mode):
    """
    Feature: Ops.
    Description: test op cumsum.
    Expectation: expect correct result.
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    ms.set_context(mode=mode)
    test_shape = (2, 3, 4, 5)
    dim1 = 0
    x, _ = generate_random_input(test_shape, dim1)
    output = cumsum_backward_func(ms.Tensor(x), dim1)
    expect = np.flip(np.cumsum(np.flip(np.ones(test_shape), dim1), dim1), dim1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

    dim2 = 2
    x, _ = generate_random_input(test_shape, dim2)
    output2 = cumsum_backward_func(ms.Tensor(x), dim2)
    expect2 = np.flip(np.cumsum(np.flip(np.ones(test_shape), dim2), dim2), dim2)
    np.testing.assert_allclose(output2.asnumpy(), expect2, rtol=1e-4)
    del os.environ['GRAPH_OP_RUN']


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_cumsum_dynamic_shape(mode):
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    os.environ['GRAPH_OP_RUN'] = '1'
    dim1 = 0
    ms_data1, _ = generate_random_input((2, 3, 4), dim1)
    dim2 = 1
    ms_data2, _ = generate_random_input((3, 4, 5, 6), dim2)
    TEST_OP(cumsum_forward_func, [[ms.Tensor(ms_data1), dim1], [ms.Tensor(ms_data2), dim2]],
            grad=True, mode=mode)
    del os.environ['GRAPH_OP_RUN']
