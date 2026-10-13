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
"""Tests for ops.hstack and its compatibility paths."""

import pytest
import numpy as np
import mindspore as ms
from mindspore import ops
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(inputs):
    return np.hstack(inputs)

def generate_expect_backward_output(inputs):
    return [np.ones_like(input_x) for input_x in inputs]

@test_utils.run_with_cell
def hstack_forward_func(tensor1, tensor2):
    return ops.hstack((tensor1, tensor2))

@test_utils.run_with_cell
def hstack_backward_func(tensor1, tensor2):
    input_grad = ms.grad(hstack_forward_func, (0, 1))(tensor1, tensor2)
    return input_grad

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_hstack_normal(mode):
    """
    Feature: ops.hstack
    Description: verify the result of hstack
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([np.random.randn(), np.random.randn()])
    input_list.append([generate_random_input((2,), np.int64), generate_random_input((5,), np.int64)])
    input_list.append([generate_random_input((2, 2), np.float16), generate_random_input((2, 3), np.float16)])
    input_list.append([generate_random_input((2, 1, 4), np.float32), generate_random_input((2, 3, 4), np.float32)])
    input_list.append([generate_random_input((2, 3, 4, 5), np.int32), generate_random_input((2, 3, 4, 5), np.int32)])
    input_list.append([generate_random_input((2, 3, 4), np.complex64), generate_random_input((2, 2, 4), np.complex64)])

    for inputs in input_list:
        tensors = [ms.Tensor(input_x) for input_x in inputs]
        expect = generate_expect_forward_output(inputs)
        output = hstack_forward_func(*tensors)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
        if np.asarray(inputs[0]).dtype.kind in ("f", "c"):
            expect_grads = generate_expect_backward_output(inputs)
            output_grads = hstack_backward_func(*tensors)
            assert len(output_grads) == len(expect_grads)
            for output_grad, expect_grad in zip(output_grads, expect_grads):
                np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', mem_peak=2048.00)
def test_ops_hstack_dynamic():
    """
    Feature: ops.hstack
    Description: test function hstack_forward_func with dynamic shape and dynamic rank
    Expectation: success
    """
    inputs1 = [generate_random_input((5, 4, 3, 2), np.float32), generate_random_input((5, 4, 3, 2), np.float32)]
    inputs2 = [generate_random_input((2, 3, 2), np.float32), generate_random_input((2, 3, 2), np.float32)]
    TEST_OP(
        hstack_forward_func,
        [[ms.Tensor(inputs1[0]), ms.Tensor(inputs1[1])], [ms.Tensor(inputs2[0]), ms.Tensor(inputs2[1])]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('shape', [(3,), (2, 3)])
def test_ops_hstack_cpu_graph(shape):
    """
    Feature: ops.hstack CPU graph compatibility.
    Description: Exercise the existing ops concatenation path.
    Expectation: Values, shape and dtype match NumPy horizontal stacking.
    """
    ms.set_context(mode=ms.GRAPH_MODE, device_target="CPU")
    x = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    y = x + 10
    output = hstack_forward_func(ms.Tensor(x), ms.Tensor(y))
    expected = np.hstack((x, y))
    assert output.shape == expected.shape
    assert output.dtype == ms.float32
    np.testing.assert_array_equal(output.asnumpy(), expected)
