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
import mindspore.nn as nn
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark

def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)

class UnbindNet(nn.Cell):
    def construct(self, input_x, dim=0):
        return input_x.unbind(dim)

def ops_unbind_compare(output_tuple, expect_tuple):
    assert len(output_tuple) == len(expect_tuple)
    for output, expect in zip(output_tuple, expect_tuple):
        np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-4)

def ops_unbind_float32_1d_case():
    x_np = generate_random_input((2,), np.float32)
    input_x = ms.Tensor(x_np)
    dim = 0

    net = UnbindNet()
    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[i])
    output_tuple = net(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

def ops_unbind_float32_5d_case():
    x_np = generate_random_input((2, 3, 4, 5, 6), np.float32)
    input_x = ms.Tensor(x_np)
    dim = 3

    net = UnbindNet()
    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[:, :, :, i, :])
    output_tuple = net(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

def ops_unbind_float32_dim_neg_case():
    x_np = generate_random_input((2, 3, 4), np.float32)
    input_x = ms.Tensor(x_np)
    dim = -2

    net = UnbindNet()
    expect_tuple = []
    for i in range(input_x.shape[dim]):
        expect_tuple.append(x_np[:, i, :])
    output_tuple = net(input_x, dim)
    ops_unbind_compare(output_tuple, expect_tuple)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['KBK'])
def test_deprecated_tensor_unbind(mode): # It will be called in KBK mode.
    """
    Feature: tensor.unbind
    Description: verify the result of the deprecated tensor.unbind
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    ops_unbind_float32_1d_case()
    ops_unbind_float32_5d_case()
    ops_unbind_float32_dim_neg_case()

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
                      'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=2.0)
@pytest.mark.parametrize('mode', ['pynative'])
def test_pyboost_tensor_unbind(mode): # It will be called only in pynative mode.
    """
    Feature: tensor.unbind
    Description: verify the result of the tensor.unbind
    Expectation: success
    """
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    elif mode == 'KBK':
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    ops_unbind_float32_1d_case()
    ops_unbind_float32_5d_case()
    ops_unbind_float32_dim_neg_case()

@arg_mark(plat_marks=['platform_ascend', 'platform_ascend910b'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2.0)
def test_tensor_unbind_dynamic():
    """
    Feature: tensor.unbind
    Description: test function tensor.unbind with dynamic shape and dynamic rank
    Expectation: success
    """
    input1 = generate_random_input((2, 3, 4, 5), np.float32)
    input2 = generate_random_input((3, 5, 2), np.float32)
    net = UnbindNet()
    TEST_OP(
        net,
        [[ms.Tensor(input1), 0], [ms.Tensor(input2), 1]],
        disable_mode=["GRAPH_MODE_GE", "GRAPH_MODE_O0"],
        disable_case=['EmptyTensor', 'ScalarTensor'],
        case_config={'disable_input_check': True}
    )
