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
import mindspore
from mindspore import Tensor
from mindspore import ops
from mindspore import mint
from mindspore import jit, JitConfig

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(input_tensor, repeats, axis=None):
    return np.repeat(input_tensor, repeats, axis)

def generate_expect_backward_output(input_tensor, repeats, axis):
    if isinstance(repeats, int):
        repeats = [repeats,]
    output = []
    if len(repeats) == 1:
        output = repeats[0]*np.ones_like(input_tensor, np.float32)
    else:
        if axis == 0:
            output = [r*np.ones_like(input_tensor[0], np.float32) for r in repeats]
        elif axis == 1:
            output = [repeats for i in range(input_tensor.shape[0])]
    return output

def repeat_interleave_forward(input_tensor, repeats, axis):
    return mint.repeat_interleave(input_tensor, repeats, axis)

def repeat_interleave_backward(input_tensor, repeats, axis):
    input_grad = ops.grad(repeat_interleave_forward)(input_tensor, repeats, axis)
    return input_grad

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [mindspore.GRAPH_MODE, mindspore.PYNATIVE_MODE])
@pytest.mark.parametrize('axis', [0, 1])
def test_repeat_interleave_forward_int(mode, axis):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of mint.repeat_interleave when `repeats` is integer
    Expectation: success
    """
    mindspore.set_context(mode=mode)
    x = generate_random_input((5, 3), np.float32)
    repeats = 2
    output = (jit(repeat_interleave_forward, jit_config=JitConfig(jit_level="O0")))(Tensor(x), repeats, axis)
    expect = generate_expect_forward_output(x, repeats, axis)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [mindspore.GRAPH_MODE, mindspore.PYNATIVE_MODE])
@pytest.mark.parametrize('axis', [0, 1])
def test_repeat_interleave_backward_int(mode, axis):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of back propagation for mint.repeat_interleave when `repeats` is integer
    Expectation: success
    """
    mindspore.set_context(mode=mode)
    x = generate_random_input((5, 3), np.float32)
    repeats = 2
    output = (jit(repeat_interleave_backward, jit_config=JitConfig(jit_level="O0")))(Tensor(x), repeats, axis)
    expect = generate_expect_backward_output(x, repeats, axis)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [mindspore.GRAPH_MODE, mindspore.PYNATIVE_MODE])
@pytest.mark.parametrize('axis', [0, 1])
def test_repeat_interleave_forward_tensor(mode, axis):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of mint.repeat_interleave when `repeats` is tensor
    Expectation: success
    """
    mindspore.set_context(mode=mode)
    x = generate_random_input((5, 3), np.float32)
    repeats = [np.random.randint(1, 5) for i in range(x.shape[axis])]
    output = (jit(repeat_interleave_forward, jit_config=JitConfig(jit_level="O0")))(Tensor(x), repeats, axis)
    expect = generate_expect_forward_output(x, repeats, axis)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)

@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [mindspore.PYNATIVE_MODE, mindspore.GRAPH_MODE])
@pytest.mark.parametrize('axis', [0, 1])
def test_repeat_interleave_backward_tensor(mode, axis):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of back propagation for mint.repeat_interleave when `repeats` is tensor
    Expectation: success
    """
    mindspore.set_context(mode=mode)
    x = generate_random_input((5, 3), np.float32)
    repeats = [np.random.randint(1, 5) for i in range(x.shape[axis])]
    output = (jit(repeat_interleave_backward, jit_config=JitConfig(jit_level="O0")))(Tensor(x), repeats, axis)
    expect = generate_expect_backward_output(x, repeats, axis)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3)
