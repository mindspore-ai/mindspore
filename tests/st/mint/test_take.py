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
from mindspore import ops, mint, jit, JitConfig
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


class Net(ms.nn.Cell):
    def construct(self, input_x, index):
        out = mint.take(input_x, index)
        return out


def generate_random_input(shape, dtype):
    return np.random.uniform(-1, 1, shape).astype(dtype)


def generate_expect_output(input_x, index):
    reshape_input = input_x.reshape(24)
    out = []
    for item in index:
        out.append(reshape_input[item])
    return ms.Tensor(out)


def mint_take_forward_func(input_x, index):
    return Net()(input_x, index)


def mint_take_backward_func(input_x, index):
    net = Net()
    grad = ops.GradOperation(get_all=True)
    return grad(net)(input_x, index)


def mint_take_forward_func_tensor(input_x, index):
    return input_x.take(index)


def mint_take_backward_func_tensor(input_x, index):
    net = mint_take_forward_func_tensor
    grad = ops.GradOperation(get_all=True)
    return grad(net)(input_x, index)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_take_std(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    input_x = generate_random_input((2, 3, 4), np.float32)
    index = [3, 4]

    expect_out = generate_expect_output(input_x, index)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = mint_take_forward_func(ms.Tensor(input_x), ms.Tensor(index))
    else:
        output = (jit(mint_take_forward_func, backend="ms_backend", jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input_x), ms.Tensor(index))
    np.allclose(output.asnumpy(), expect_out.asnumpy(),
                rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2.00)
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_take_tensor(mode):
    """
    Feature: standard forward, backward features.
    Description: test function copy.
    Expectation: expect correct result.
    """
    input_x = generate_random_input((2, 3, 4), np.float32)
    index = [3, 4]

    expect_out = generate_expect_output(input_x, index)
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
        output = mint_take_forward_func_tensor(ms.Tensor(input_x), ms.Tensor(index))
    else:
        output = (jit(mint_take_forward_func_tensor, jit_config=JitConfig(jit_level="O0")))(
            ms.Tensor(input_x), ms.Tensor(index))
    np.allclose(output.asnumpy(), expect_out.asnumpy(),
                rtol=1e-5, equal_nan=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_take_dynamic_shape():
    """
    Feature: dynamic shape forward, backward features.
    Description: test copy forward with dynamic shape.
    Expectation: expect correct result.
    """
    tensor_x1 = ms.Tensor(generate_random_input((2, 3), np.float32))
    tensor_x2 = ms.Tensor(generate_random_input((3, 4, 5), np.float32))
    tensor_y1 = ms.Tensor([2, 3], ms.int64)
    tensor_y2 = ms.Tensor([[2, 3], [4, 5]], ms.int64)

    TEST_OP(mint_take_forward_func, [[tensor_x1, tensor_y1], [tensor_x2, tensor_y2]],
            disable_mode=['GRAPH_MODE_GE'])
