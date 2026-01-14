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
from mindspore import mint
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

def generate_expect_forward_output(inputs):
    return np.dstack(inputs)

def generate_expect_backward_output(inputs):
    return [np.ones_like(input_x) for input_x in inputs]

@test_utils.run_with_cell
def dstack_forward_func(tensor1, tensor2):
    return mint.dstack((tensor1, tensor2))

@test_utils.run_with_cell
def dstack_backward_func(tensor1, tensor2):
    input_grad = ms.grad(dstack_forward_func, (0, 1))(tensor1, tensor2)
    return input_grad

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_ops_dstack_normal(mode):
    """
    Feature: mint.dstack
    Description: verify the result of dstack
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    input_list = []
    input_list.append([np.random.randn(), np.random.randn()])
    input_list.append([generate_random_input((2,), np.int64), generate_random_input((2,), np.int64)])
    input_list.append([generate_random_input((2, 3), np.float16), generate_random_input((2, 3), np.float16)])
    input_list.append([generate_random_input((2, 3, 1), np.float32), generate_random_input((2, 3, 4), np.float32)])
    input_list.append([generate_random_input((2, 3, 4, 5), np.int32), generate_random_input((2, 3, 4, 5), np.int32)])
    input_list.append([generate_random_input((2, 3, 4), np.complex64), generate_random_input((2, 3, 2), np.complex64)])

    for i in range(len(input_list)):
        inputs = input_list[i]
        tensors = [ms.Tensor(input_x) for input_x in inputs]
        expect = generate_expect_forward_output(inputs)
        output = dstack_forward_func(*tensors)
        expect_grads = generate_expect_backward_output(inputs)
        output_grads = dstack_backward_func(*tensors)
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)
        assert len(output_grads) == len(expect_grads)
        for output_grad, expect_grad in zip(output_grads, expect_grads):
            np.testing.assert_allclose(output_grad.asnumpy(), expect_grad, rtol=1e-4)

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=4.0)
def test_ops_dstack_dynamic():
    """
    Feature: mint.dstack
    Description: test function dstack_forward_func with dynamic shape and dynamic rank
    Expectation: success
    """
    inputs1 = [generate_random_input((2, 3, 2, 5), np.float32), generate_random_input((2, 3, 2, 5), np.float32)]
    inputs2 = [generate_random_input((3, 5, 2), np.float32), generate_random_input((3, 5, 2), np.float32)]
    TEST_OP(
        dstack_forward_func,
        [[ms.Tensor(inputs1[0]), ms.Tensor(inputs1[1])], [ms.Tensor(inputs2[0]), ms.Tensor(inputs2[1])]],
        disable_mode=["GRAPH_MODE_GE"],
        case_config={'disable_input_check': True,
                     'all_dim_zero': True}
    )
