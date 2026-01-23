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
"""
Tests for mint.repeat_interleave.
"""
import pytest
import numpy as np
import mindspore as ms
from mindspore import Tensor
from mindspore import ops, mint
from mindspore import jit
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.st.utils import test_utils
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(input_tensor, repeats, dim=None):
    return np.repeat(input_tensor, repeats, dim)


def generate_expect_backward_output(input_tensor, repeats, dim):
    if isinstance(repeats, int):
        repeats = [repeats,]
    output = []
    if len(repeats) == 1:
        output = repeats[0] * np.ones_like(input_tensor, np.float32)
    else:
        if dim == 0:
            output = [r * np.ones_like(input_tensor[0], np.float32) for r in repeats]
        elif dim == 1 or dim == -1:
            output = [repeats for i in range(input_tensor.shape[0])]
        elif dim is None:
            output = np.reshape(repeats, input_tensor.shape)
    return output


@test_utils.run_with_cell
def repeat_interleave_forward(input_tensor, repeats, dim, output_size=None):
    return mint.repeat_interleave(input_tensor, repeats, dim, output_size=output_size)


@test_utils.run_with_cell
def repeat_interleave_backward(input_tensor, repeats, dim, output_size=None):
    input_grad = ops.grad(repeat_interleave_forward)(input_tensor, repeats, dim, output_size=output_size)
    return input_grad


def set_mode(mode):
    if mode.lower() == 'kbk':
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    else:
        ms.set_context(mode=ms.PYNATIVE_MODE)


def repeat_interleave_case(x_np, repeats, dim, cast_tensor=False):
    loss = 0
    if x_np.dtype == np.float16:
        loss = 1e-3
    elif x_np.dtype == np.float32:
        loss = 1e-4
    elif x_np.dtype == np.bfloat16:
        loss = 4e-3

    expect = generate_expect_forward_output(x_np, repeats, dim)
    expect_grad = generate_expect_backward_output(x_np, repeats, dim)

    if cast_tensor and isinstance(repeats, list):
        repeats = Tensor(repeats, dtype=ms.int64)

    output = repeat_interleave_forward(Tensor(x_np), repeats, dim)
    output_grad = repeat_interleave_backward(Tensor(x_np), repeats, dim)

    np.testing.assert_allclose(output.asnumpy(), expect, rtol=loss)
    np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=loss)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dim', [0, -1, None])
def test_repeat_interleave_forward_backward_int(mode, dim):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of mint.repeat_interleave when `repeats` is integer
    Expectation: success
    """
    set_mode(mode)

    x = generate_random_input((5, 3), np.float32)
    repeats = 2
    repeat_interleave_case(x, repeats, dim)

    repeats = 0
    repeat_interleave_case(x, repeats, dim)


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential', mem_peak=20.01)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dim', [0, None])
@pytest.mark.parametrize('cast_tensor', [True, False])
def test_repeat_interleave_forward_backward_tensor(mode, dim, cast_tensor):
    """
    Feature: mint.repeat_interleave
    Description: Verify the result of mint.repeat_interleave when `repeats` is tensor
    Expectation: success
    """
    set_mode(mode)

    x = generate_random_input((4, 4), np.float32)

    if dim is None:
        repeats = np.random.randint(10, size=16).tolist()
    else:
        repeats = [np.random.randint(1, 5) for i in range(x.shape[dim])]
    repeat_interleave_case(x, repeats, dim, cast_tensor)

    repeats = [4,]
    repeat_interleave_case(x, repeats, dim, cast_tensor)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('dim', [None, 1])
def test_repeat_interleave_bfloat16(mode, dim):
    """
    Feature: test ne functional API.
    Description: testcase for ne functional API.
    Expectation: the result match with expected result.
    """
    x = generate_random_input((3, 4), np.float32)
    if dim is None:
        repeats = np.random.randint(10, size=12)
    else:
        repeats = [np.random.randint(1, 5) for i in range(x.shape[dim])]
    expect = generate_expect_forward_output(x, repeats, dim)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = repeat_interleave_forward(Tensor(x, dtype=ms.bfloat16), Tensor(repeats), dim)
    else:
        output = (jit(repeat_interleave_forward, jit_level="O0"))(
            Tensor(x, dtype=ms.bfloat16), Tensor(repeats), dim)
    assert np.allclose(output.float().asnumpy(), expect, 0.004, 0.004)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=20.0)
def test_repeat_interleave_dynamic_shape_int():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    dim1 = 0
    repeats1 = 5
    input_case1 = Tensor(np.random.rand(2, 3).astype(np.float32))
    input_case2 = Tensor(np.random.rand(3, 4, 5).astype(np.float32))
    dim2 = 1
    repeats2 = 7
    TEST_OP(repeat_interleave_forward, [[input_case1, repeats1, dim1], [input_case2, repeats2, dim2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=20.0)
def test_repeat_interleave_dynamic_shape_tensor():
    """
    Feature: Test dynamic shape.
    Description: test function div dynamic feature.
    Expectation: expect correct result.
    """
    dim1 = 0
    repeats1 = Tensor([4, 2])
    output_size1 = 6
    input_case1 = Tensor(np.random.rand(2, 3).astype(np.float32))
    input_case2 = Tensor(np.random.rand(3, 4, 5).astype(np.float32))
    dim2 = 1
    output_size2 = 14
    repeats2 = Tensor([2, 3, 5, 4])
    TEST_OP(repeat_interleave_forward, [[input_case1, repeats1, dim1, output_size1],
                                        [input_case2, repeats2, dim2, output_size2]],
            disable_mode=['GRAPH_MODE_GE'],
            disable_case=['EmptyTensor',
                          'ScalarTensor'],
            case_config={'disable_input_check': True,
                         'deterministic_use_origin_inputs': True})
