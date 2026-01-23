# Copyright 2024 Huawei Technomulies Co., Ltd
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
"""Test ops mul."""
import pytest
import numpy as np
import mindspore as ms
from mindspore import ops, context, nn
from mindspore.mint import mul
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x, y):
    return np.multiply(x, y)


def generate_expect_backward_output(x, y):
    return y, x


@test_utils.run_with_cell
def mul_forward_func(x, y):
    return mul(x, y)


@test_utils.run_with_cell
def mul_backward_func(x, y):
    return ms.grad(mul_forward_func, (0, 1))(x, y)


@test_utils.run_with_cell
def mul_vmap_func(x, y):
    return ops.vmap(mul_forward_func, in_axes=0, out_axes=0)(x, y)


def set_mode(mode):
    """
    set_mode
    """
    if mode == "ge":
        context.set_context(mode=context.GRAPH_MODE, jit_config={'jit_level': 'O2'})
    elif mode == "kbk":
        context.set_context(mode=context.GRAPH_MODE, jit_config={'jit_level': 'O0'})
    else:
        context.set_context(mode=context.PYNATIVE_MODE)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2082.01)
@pytest.mark.parametrize("mode", ["ge", "kbk", "pyboost"])
def test_ops_mul_normal(mode):
    """
    Feature: pyboost function.
    Description: test function mul forward and backward.
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = generate_random_input((1, 16, 4096, 128), np.float32)
    y = generate_random_input((4096, 128), np.float32)
    output = mul_forward_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)

    x2 = generate_random_input((2, 3, 4), np.float32)
    y2 = generate_random_input((2, 3, 4), np.float32)
    output2 = mul_backward_func(ms.Tensor(x2), ms.Tensor(y2))
    expect_out2 = generate_expect_backward_output(x2, y2)
    np.testing.assert_allclose(output2[0].asnumpy(), expect_out2[0], rtol=1e-3)
    np.testing.assert_allclose(output2[1].asnumpy(), expect_out2[1], rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.00)
@pytest.mark.parametrize("mode", ["kbk", "pyboost"])
def test_ops_mul_forward_case01(mode):
    """
    Feature: pyboost function.
    Description: test function mul forward.
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = generate_random_input((3, 77, 3, 768), np.float16)
    y = 0.125
    output = mul_forward_func(ms.Tensor(x), y)
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=132.0)
@pytest.mark.parametrize("mode", ["ge", "kbk", "pyboost"])
def test_ops_mul_forward_case02(mode):
    """
    Feature: pyboost function.
    Description: test function mul forward.
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = np.random.rand(1, 1, 4096)
    y = np.random.rand(1, 4096, 4096)
    output = mul_forward_func(ms.Tensor(x, ms.bfloat16), ms.Tensor(y, ms.bfloat16))
    expect_out = np.multiply(x, y).astype(np.float32)
    np.testing.assert_allclose(output.float().asnumpy(), expect_out, rtol=4e-3, atol=4e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["ge", "kbk", "pyboost"])
def test_ops_mul_vmap(mode):
    """
    Feature: pyboost function.
    Description: test function mul vmap feature.
    Expectation: expect correct result.
    """
    set_mode(mode)
    x = generate_random_input((2, 3, 4, 5), np.float32)
    y = generate_random_input((2, 3, 4, 5), np.float32)
    output = mul_vmap_func(ms.Tensor(x), ms.Tensor(y))
    expect_out = generate_expect_forward_output(x, y)
    np.testing.assert_allclose(output.asnumpy(), expect_out, rtol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
def test_ops_mul_dynamic_shape():
    """
    Feature: pyboost function.
    Description: test function mul with dynamic shape and dynamic rank.
    Expectation: return the correct value.
    """
    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    y1 = generate_random_input((2, 3, 4, 5), np.float32)
    x2 = generate_random_input((3, 4, 5, 6, 7), np.float32)
    y2 = generate_random_input((3, 4, 5, 6, 7), np.float32)

    TEST_OP(mul_forward_func,
            [[ms.Tensor(x1), ms.Tensor(y1)], [ms.Tensor(x2), ms.Tensor(y2)]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'all_dim_zero': True})


class MulNetTensorTensor(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = ms.Tensor([1, -1, 1, 2], dtype=ms.float32)
        self.y = ms.Tensor([2, 1, 2, 2], dtype=ms.float32)

    def construct(self):
        return mul(self.x, self.y)


class MulNetTensorScalar(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = ms.Tensor([1, -1, 1, 2], dtype=ms.float32)

    def construct(self):
        return mul(self.x, 1)


class MulNetScalarTensor(nn.Cell):
    def __init__(self):
        super().__init__()
        self.x = ms.Tensor([1, -1, 1, 2], dtype=ms.float32)

    def construct(self):
        return mul(1, self.x)


class MulNetScalarScalar(nn.Cell):
    def construct(self):
        return mul(2, 4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize("mode", ["kbk"])
def test_mul_infer_value(mode):
    """
    Feature: Mul for infer value
    Description: test ops mul
    Expectation: expect correct result.
    """
    set_mode(mode)

    net1 = MulNetTensorTensor()
    out1 = net1()
    assert np.allclose(out1.asnumpy(), np.array([2, -1, 2, 4]), 0.0001, 0.0001)

    net2 = MulNetTensorScalar()
    out2 = net2()
    assert np.allclose(out2.asnumpy(), np.array([1, -1, 1, 2]), 0.0001, 0.0001)

    net3 = MulNetScalarTensor()
    out3 = net3()
    assert np.allclose(out3.asnumpy(), np.array([1, -1, 1, 2]), 0.0001, 0.0001)

    net4 = MulNetScalarScalar()
    out4 = net4()
    assert out4.asnumpy() == 8
