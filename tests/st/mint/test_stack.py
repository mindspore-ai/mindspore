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
from tests.mark_utils import arg_mark

from functools import reduce
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from mindspore.mint import stack
from mindspore import jit, JitConfig
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


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
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


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
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
