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
import numpy as np
import pytest

import mindspore as ms
from mindspore import ops
from mindspore.scipy.fft import idct, dct
from scipy.fft import idct as sp_idct
from scipy.fft import dct as sp_dct

import tests.st.utils.test_utils as test_utils
from tests.mark_utils import arg_mark

ms.context.set_context(device_target="CPU")


# idct functional api test cases
def idct_generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def idct_generate_expect_forward_output(x):
    return sp_idct(x, norm='ortho').astype(np.float32)


def idct_generate_expect_backward_output(x):
    # out = sp_dct(x, norm='ortho')
    out = np.array([[1.73205081, 0., 0.], [1.73205081, 0., 0.]], dtype=np.float32)
    return out


def idct_generate_expect_backward_output_3_4(x):
    # out = sp_dct(x, norm='ortho')
    out = np.array([[2., 0., 0., 0.],
                    [2., 0., 0., 0.],
                    [2., 0., 0., 0.]], dtype=np.float32)
    return out


@test_utils.run_with_cell
def idct_forward_func(x):
    return idct(x)


@test_utils.run_with_cell
def idct_backward_func(x):
    return ops.grad(idct_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = idct_generate_random_input((2, 3), np.float32)
    output = idct_forward_func(ms.Tensor(x))
    expect = idct_generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = idct_generate_random_input((2, 3), np.float32)
    output = idct_backward_func(ms.Tensor(x))
    ones_grad = ms.Tensor(np.ones_like(x))
    expect = idct_generate_expect_backward_output(ones_grad)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(idct_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = idct_generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = idct_generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = idct_generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = idct_generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(idct_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = idct_generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = idct_generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = idct_generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = idct_generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(idct_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = idct_generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = idct_generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = idct_generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = idct_generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(idct_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = idct_generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = idct_generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = idct_generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = idct_generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


# dct functional api test cases
def dct_generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def dct_generate_expect_forward_output(x):
    return sp_dct(x).astype(np.float32)


def dct_generate_expect_backward_output(x):
    out = np.array([[4.732051, 0, 1.2679492], [4.732051, 0, 1.2679492]], dtype=np.float32)
    return out


def dct_generate_expect_backward_output_3_4(x):
    out = np.array([[6.0273395, -0.49660575, 1.6681787, 0.8010876],
                    [6.0273395, -0.49660575, 1.6681787, 0.8010876],
                    [6.0273395, -0.49660575, 1.6681787, 0.8010876]], dtype=np.float32)
    return out


@test_utils.run_with_cell
def dct_forward_func(x):
    return dct(x)


@test_utils.run_with_cell
def dct_backward_func(x):
    return ops.grad(dct_forward_func, (0))(x)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = dct_generate_random_input((2, 3), np.float32)
    output = dct_forward_func(ms.Tensor(x))
    expect = dct_generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = dct_generate_random_input((2, 3), np.float32)
    output = dct_backward_func(ms.Tensor(x))
    ones_grad = ms.Tensor(np.ones_like(x))
    expect = dct_generate_expect_backward_output(ones_grad)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None, None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(dct_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = dct_generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = dct_generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = dct_generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = dct_generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_forward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct forward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(dct_forward_func)
    test_cell.set_inputs(x_dyn)

    x1 = dct_generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = dct_generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = dct_generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = dct_generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward_dynamic_shape(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic shape.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=[None, None], dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(dct_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = dct_generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = dct_generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = dct_generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = dct_generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@arg_mark(plat_marks=['platform_ascend', 'cpu_linux', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_dct_backward_dynamic_rank(mode):
    """
    Feature: mindspore.scipy.fft.dct
    Description: test function dct backward with dynamic rank.
    Expectation: success
    """
    ms.context.set_context(mode=mode)

    x_dyn = ms.Tensor(shape=None, dtype=ms.float32)
    test_cell = test_utils.to_cell_obj(dct_backward_func)
    test_cell.set_inputs(x_dyn)

    x1 = dct_generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = dct_generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = dct_generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = dct_generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)
