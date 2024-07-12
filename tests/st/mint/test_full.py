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
# pylint: disable=unused-variable
import pytest
import numpy as np
import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore import ops, mint, Tensor, jit, JitConfig, context, nn
from tests.st.ops.dynamic_shape.test_op_utils import TEST_OP


class FullNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.full = mint.full

    def construct(self, size, fill_value, dtype):
        return self.full(size, fill_value, dtype=dtype)

class FullGradNet(nn.Cell):
    def __init__(self):
        super().__init__()
        self.grad_func = ops.GradOperation(get_all=True)

    def wrap_full(self, size, fill_value, dtype):
        return mint.full(size, fill_value, dtype=dtype)

    def construct(self, size, fill_value, dtype):
        full_grad = self.grad_func(self.wrap_full)
        return full_grad(size, fill_value, dtype)[0]


def full_forward_func(size, fill_value, dtype=None):
    y = mint.full(size, fill_value, dtype=dtype)
    return y

def full_backward_func(size, fill_value, dtype=None):
    value_grad = ops.grad(full_forward_func, (1,))(size, fill_value, dtype=dtype)
    return value_grad


@pytest.mark.level0
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize('mode', ['GE', 'pynative', 'KBK'])
def test_full_forward_backward(mode):
    """
    Feature: Ops.
    Description: test full.
    Expectation: expect correct result.
    """
    size = (1, 2, 3)
    value = 6.
    dtype = None
    expect_y = np.array([[[6., 6., 6.],
                          [6., 6., 6.]]])
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        y = full_forward_func(size, value, dtype)
    elif mode == 'KBK':
        y = (jit(full_forward_func, jit_config=JitConfig(jit_level="O0")))(size, value, dtype)
    else:
        y = (jit(full_forward_func, jit_config=JitConfig(jit_level="O2")))(size, value, dtype)
    np.testing.assert_allclose(y.asnumpy(), expect_y, rtol=1e-5)

    value = Tensor(6)
    dtype = mstype.int32
    expect_value_grad = 0
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        value_grad = full_backward_func(size, value, dtype)
    elif mode == 'KBK':
        value_grad = (jit(full_backward_func, jit_config=JitConfig(jit_level="O0")))(size, value, dtype)
    else:
        value_grad = (jit(full_backward_func, jit_config=JitConfig(jit_level="O2")))(size, value, dtype)
    np.testing.assert_allclose(value_grad.asnumpy(), expect_value_grad, rtol=1e-5)
    assert value_grad.shape == ()


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
def test_full_dynamic_shape():
    """
    Feature: Test full with dynamic shape in graph mode.
    Description: call ops.mint.full with valid input and index.
    Expectation: return the correct value.
    """
    size_1 = (1, 2, 3)
    value_1 = Tensor([5])
    size_2 = (4, 3, 2)
    value_2 = Tensor(6)
    TEST_OP(full_forward_func, [[size_1, value_1], [size_2, value_2]], '', disable_input_check=True,
            disable_yaml_check=True)

    size_1 = (1, 2, 3)
    value_1 = 5
    size_2 = (4, 3, 2)
    value_2 = 6
    TEST_OP(full_forward_func, [[size_1, value_1], [size_2, value_2]], '', disable_input_check=True,
            disable_yaml_check=True, disable_grad=True)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_full_forward_dynamic_rank(context_mode):
    """
    Feature: full ops.
    Description: test ops full with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size = (2, 3)
    value_dyn = Tensor(shape=None, dtype=mstype.float32)
    test_cell = FullNet()
    test_cell.set_inputs(size, value_dyn, ms.int32)
    value = Tensor(np.array([3]).astype(np.float32))
    out = test_cell(size, value, ms.int32)
    expect_output = np.full((2, 3), 3, np.int32)
    assert np.allclose(out.asnumpy(), expect_output)

    with pytest.raises(ValueError):
        value = Tensor(np.array([3, 4]).astype(np.float32))
        _ = test_cell(size, value, ms.int32)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.parametrize("context_mode", [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_full_backward_dynamic_rank(context_mode):
    """
    Feature: full ops.
    Description: test ops full with dynamic shape tensor input.
    Expectation: output the right result.
    """
    context.set_context(mode=context_mode)
    size = (2, 3)
    value_dyn = Tensor(shape=None, dtype=mstype.float32)
    test_cell = FullGradNet()
    test_cell.set_inputs(size, value_dyn, ms.int32)
    value = Tensor(np.array([3]).astype(np.float32))
    dvalue = test_cell(size, value, ms.int32)
    expect_dvalue = [0]
    assert np.allclose(dvalue.asnumpy(), expect_dvalue)

    # The forward graph will be remove by optimization pass in GRAPH_MODE, since the input value is not used during
    # backward. This will likely cause input validation loss in dynamic scene.
    if context_mode == ms.PYNATIVE_MODE:
        with pytest.raises(ValueError):
            value = Tensor(np.array([2, 3]).astype(np.float32))
            _ = test_cell(size, value, ms.int32)
