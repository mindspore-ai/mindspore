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
from mindspore import ops, Tensor

import tests.st.utils.test_utils as test_utils
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


@test_utils.run_with_cell
def norm_ext_forward_func(x):
    return ops.norm_ext(x)

@test_utils.run_with_cell
def norm_ext_backward_func(x):
    return ops.grad(norm_ext_forward_func, (0))(x)



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_norm_forward(mode):
    """
    Feature: norm
    Description: Verify the result of norm
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    a = ops.arange(9, dtype=ms.float32) - 4
    b = a.reshape((3, 3))
    output1 = norm_ext_forward_func(a)
    expect_output1 = np.array(7.745967)
    assert np.allclose(output1.asnumpy(), expect_output1)

    output2 = norm_ext_forward_func(b)
    expect_output2 = np.array(7.745967)
    assert np.allclose(output2.asnumpy(), expect_output2)



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_norm_backward(mode):
    """
    Feature: norm backward
    Description: Verify the result of norm backward
    Expectation: success
    """
    ms.set_context(jit_level='O0')
    ms.set_context(mode=mode)
    a = ops.arange(9, dtype=ms.float32) - 4
    b = a.reshape((3, 3))
    output1 = norm_ext_backward_func(a)
    expect_output1 = ops.grad(ops.norm, (0))(a).asnumpy()
    assert np.allclose(output1.asnumpy(), expect_output1)

    output2 = norm_ext_backward_func(b)
    expect_output2 = ops.grad(ops.norm, (0))(b).asnumpy()
    assert np.allclose(output2.asnumpy(), expect_output2)



@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_ops_norm_dyn():
    """
    Feature: pyboost function.
    Description: test Norm with dynamic rank/shape.
    Expectation: success.
    """
    input_x1 = np.random.randn(*(3, 3)).astype(np.float32)
    input_x2 = np.random.randn(*(3, 3, 3)).astype(np.float32)
    in1 = Tensor(input_x1)
    in2 = Tensor(input_x2)
    TEST_OP(norm_ext_forward_func, [[in1], [in2]], '', disable_yaml_check=True, disable_mode=['GRAPH_MODE'])
