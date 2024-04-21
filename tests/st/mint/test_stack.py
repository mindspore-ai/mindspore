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

from functools import reduce
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, mutable
from mindspore.mint import stack
from mindspore import jit, JitConfig
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP
from tests.st.utils import test_utils


def stack_func(x1, x2, axis):
    return stack((x1, x2), axis)


@test_utils.run_with_cell
def stack_forward_func(x1, x2, axis=2):
    return stack_func(x1, x2, axis)


def stack_bwd_func(x1, x2, axis):
    return ops.grad(stack_func, (0, 1))(x1, x2, axis)


@test_utils.run_with_cell
def stack_backward_func(x1, x2, axis=2):
    return stack_bwd_func(x1, x2, axis)


def stack_fwd_data_prepare(shape, axis=2):
    num = reduce(lambda x, y: x * y, shape)
    x1 = np.array([0] * num).reshape(shape).astype(np.float16)
    x2 = np.arange(num).reshape(shape).astype(np.float16)
    tensor_inputs = (ms.Tensor(x1), ms.Tensor(x2))
    expect = np.stack((x1, x2), axis)
    return tensor_inputs, expect


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_stack_forward_backward(mode):
    """
    Feature: Ops.
    Description: test op stack.
    Expectation: expect correct result.
    """
    test_shape = (2, 2, 2, 2)
    tensor_inputs, expect = stack_fwd_data_prepare(test_shape)
    expects = (np.ones(test_shape).astype(np.float16), np.ones(test_shape).astype(np.float16))
    output = stack_forward_func(tensor_inputs[0], tensor_inputs[1])
    outputs = stack_backward_func(tensor_inputs[0], tensor_inputs[1])

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
        output = stack_forward_func(tensor_inputs[0], tensor_inputs[1])
        outputs = stack_backward_func(tensor_inputs[0], tensor_inputs[1])
    else:
        output = (jit(stack_forward_func, jit_config=JitConfig(jit_level="O0")))(tensor_inputs[0], tensor_inputs[1])
        outputs = (jit(stack_backward_func, jit_config=JitConfig(jit_level="O0")))(tensor_inputs[0], tensor_inputs[1])
    assert np.allclose(output.asnumpy(), expect)
    for output, expect in zip(outputs, expects):
        assert np.allclose(output.asnumpy(), expect)


@pytest.mark.level1
@pytest.mark.platform_arm_ascend910b_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE])
def test_stack_bfloat16(mode):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    ms.set_context(mode=mode, device_target="Ascend")
    test_shape = (2, 3, 4)
    tensor_inputs, expect = stack_fwd_data_prepare(test_shape)
    output = stack_forward_func(tensor_inputs[0], tensor_inputs[1])
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('jit_level', ["O0", "O2"])
def test_stack_dynamic_shape(jit_level):
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    axis = 0
    axis_dyn = mutable(axis)
    ms_x0 = ms.Tensor(np.random.rand(2, 6, 8), ms.float32)
    ms_y0 = ms.Tensor(np.random.rand(2, 6, 8), ms.float32)
    ms_x1 = ms.Tensor(np.random.rand(3, 4, 5), ms.float32)
    ms_y1 = ms.Tensor(np.random.rand(3, 4, 5), ms.float32)
    TEST_OP(stack_forward_func, [[ms_x0, ms_y0, axis_dyn], [ms_x1, ms_y1, axis_dyn]], grad=True, jit_level=jit_level)
