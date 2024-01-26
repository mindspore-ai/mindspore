# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore.scipy.fft import idct
from scipy.fft import idct as sp_idct

import tests.st.ops.test_utils as test_utils

ms.context.set_context(device_target="CPU")


def generate_random_input(shape, dtype):
    return np.random.randn(*shape).astype(dtype)


def generate_expect_forward_output(x):
    return sp_idct(x, norm='ortho').astype(np.float32)


def generate_expect_backward_output(x):
    # out = sp_dct(x, norm='ortho')
    out = np.array([[1.73205081, 0., 0.], [1.73205081, 0., 0.]], dtype=np.float32)
    return out


def generate_expect_backward_output_3_4(x):
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


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_forward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct forward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    output = idct_forward_func(ms.Tensor(x))
    expect = generate_expect_forward_output(x)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_ops_idct_backward(mode):
    """
    Feature: mindspore.scipy.fft.idct
    Description: test function idct backward.
    Expectation: success
    """
    ms.context.set_context(mode=mode)
    x = generate_random_input((2, 3), np.float32)
    output = idct_backward_func(ms.Tensor(x))
    ones_grad = ms.Tensor(np.ones_like(x))
    expect = generate_expect_backward_output(ones_grad)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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

    x1 = generate_random_input((2, 3, 4, 5), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_forward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4, 5, 6), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_forward_output(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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

    x1 = generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.platform_arm_cpu
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
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

    x1 = generate_random_input((2, 3), np.float32)
    output = test_cell(ms.Tensor(x1))
    expect = generate_expect_backward_output(x1)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)

    x2 = generate_random_input((3, 4), np.float32)
    output = test_cell(ms.Tensor(x2))
    expect = generate_expect_backward_output_3_4(x2)
    np.testing.assert_allclose(output.asnumpy(), expect, rtol=1e-3, atol=1e-3)
