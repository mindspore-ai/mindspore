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
from mindspore import mint, nn
from mindspore.graph.api import _pynative_executor
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark


def generate_ones_input(shape, dtype):
    return np.ones(shape, dtype=dtype)


@test_utils.run_with_cell
def dropout2d_forward_func(x, p, training, inplace):
    return mint.nn.functional.dropout2d(x, p, training, inplace)


@test_utils.run_with_cell
def dropout2d_forward_func_grad(x, p, training, inplace):
    x = x * 1
    return mint.nn.functional.dropout2d(x, p, training, inplace)


@test_utils.run_with_cell
def dropout2d_backward_func(x, p, training, inplace):
    return ms.grad(dropout2d_forward_func_grad, (0,))(x, p, training, inplace)


class Dropout2d_nn(nn.Cell):
    def __init__(self, p, inplace):
        super().__init__()
        self.net = mint.nn.Dropout2d(p, inplace)
        self.net.set_train()

    def construct(self, x):
        return self.net(x)


@test_utils.run_with_cell
def dropout2d_forward_nn(x, p, inplace):
    net = Dropout2d_nn(p, inplace)
    return net(x)


@test_utils.run_with_cell
def dropout2d_forward_nn_grad(x, p, inplace):
    net = Dropout2d_nn(p, inplace)
    x = x * 1
    return net(x)


@test_utils.run_with_cell
def dropout2d_backward_nn(x, p, inplace):
    return ms.grad(dropout2d_forward_nn_grad, (0,))(x, p, inplace)


def compare_output(x, p, output):
    keep_prob = 1 - p
    if output.dtype == ms.bfloat16:
        output_np = output.float().asnumpy()
    else:
        output_np = output.asnumpy()
    elem_count = x.size
    keep_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.02)) < keep_count < (elem_count * (keep_prob + 0.02))

    expect_sum = np.array(keep_count / (1 - p), dtype=np.float64)
    output_sum = np.sum(output_np.astype(np.float64))

    if output.dtype == ms.bfloat16:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-2)
    else:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-3)


def compare_grad(x, p, grad):
    keep_prob = 1 - p
    if grad.dtype == ms.bfloat16:
        grad_np = grad.float().asnumpy()
    else:
        grad_np = grad.asnumpy()
    elem_count = x.size
    keep_count = np.count_nonzero(grad_np)
    assert (elem_count * (keep_prob - 0.02)) < keep_count < (elem_count * (keep_prob + 0.02))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard',
        essential_mark='essential', mem_peak=2672.00)
@pytest.mark.parametrize('shape', [(1280, 128), (3, 4096, 1280), (100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('inplace', [True, False])
def test_func_dropout2d(shape, mode, inplace):
    """
    Feature: standard forward, backward features.
    Description: test function dropout2d.
    Expectation: expect correct result.
    """
    x_np = generate_ones_input(shape, np.float32)
    x = ms.Tensor(x_np)
    p = 0.3
    training = True
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output = dropout2d_forward_func(x, p, training, inplace)
    assert output.shape == shape
    assert output.dtype == ms.float32
    compare_output(x_np, p, output)
    if inplace:
        compare_output(x_np, p, x)

    x1_np = generate_ones_input(shape, np.float32)
    x1 = ms.Tensor(x1_np)
    p = 0.7
    training = True
    grad = dropout2d_backward_func(x1, p, training, inplace)
    assert grad.shape == shape
    assert grad.dtype == ms.float32
    compare_grad(x1_np, p, grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=5120.01)
@pytest.mark.parametrize('shape', [(1280, 128), (3, 4096, 1280), (100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
@pytest.mark.parametrize('inplace', [True, False])
def test_nn_Dropout2d(shape, mode, inplace):
    """
    Feature: standard forward, backward features.
    Description: test function Dropout2d.
    Expectation: expect correct result.
    """
    x_np = generate_ones_input(shape, np.float32)
    x = ms.Tensor(x_np)
    p = 0.3
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output = dropout2d_forward_nn(x, p, inplace)
    assert output.shape == shape
    assert output.dtype == ms.float32
    compare_output(x_np, p, output)
    if inplace:
        compare_output(x_np, p, x)

    x1_np = generate_ones_input(shape, np.float32)
    x1 = ms.Tensor(x1_np)
    p = 0.7
    grad = dropout2d_backward_nn(x1, p, inplace)
    assert grad.shape == shape
    assert grad.dtype == ms.float32
    compare_grad(x1_np, p, grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=1336.0)
@pytest.mark.parametrize('shape', [(100, 100, 100, 100)])
@pytest.mark.parametrize('mode', ['pynative', 'KBK'])
def test_func_dropout2d_bfloat16(shape, mode):
    """
    Feature: test dropout2d functional API.
    Description: testcase for dropout2d functional API.
    Expectation: the result match with expected result.
    """
    x = generate_ones_input(shape, np.float32)
    p = 0.3
    training = True
    inplace = False
    if mode == 'pynative':
        ms.context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
    output = dropout2d_forward_func(ms.Tensor(x, dtype=ms.bfloat16), p, training, inplace)
    output_grad = dropout2d_backward_func(ms.Tensor(x, dtype=ms.bfloat16), p, training, inplace)

    assert output.shape == shape
    assert output.dtype == ms.bfloat16
    assert output_grad.shape == shape
    assert output_grad.dtype == ms.bfloat16
    compare_output(x, p, output)
    compare_grad(x, p, output_grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
def test_func_dropout2d_dynamic():
    """
    Feature: mint.nn.functional.dropout2d
    Description: dynamic
    Expectation: success
    """
    state = ms.get_rng_state()
    @test_utils.run_with_cell
    def dropout2d_func(x, p, inplace):
        ms.set_rng_state(state)
        y = x * 1
        return mint.nn.functional.dropout2d(y, p, inplace=inplace)

    x1 = ms.Tensor(generate_ones_input((2, 3, 4), np.float32))
    x2 = ms.Tensor(generate_ones_input((2, 3, 4, 5), np.float32))
    p1 = 0.3
    p2 = 0.7

    TEST_OP(dropout2d_func,
            [[x1, p1, False], [x2, p2, True]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'])

    TEST_OP(dropout2d_func,
            [[x1, p1, True], [x2, p2, False]],
            disable_mode=["GRAPH_MODE_GE"],
            disable_case=['ScalarTensor'])


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_func_dropout2d_rng_state(mode, inplace):
    """
    Feature: mint.nn.functional.dropout2d
    Description: test function with random status.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    p = 0.3

    state = ms.get_rng_state()
    x = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out1 = dropout2d_forward_func(x, p, True, inplace)

    y = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out2 = dropout2d_forward_func(y, p, True, inplace)

    ms.set_rng_state(state)
    z = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out3 = dropout2d_forward_func(z, p, True, inplace)

    if inplace:
        assert not (x.asnumpy() == y.asnumpy()).all()
        assert (x.asnumpy() == z.asnumpy()).all()
    else:
        assert not (out1.asnumpy() == out2.asnumpy()).all()
        assert (out1.asnumpy() == out3.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1',
        card_mark='onecard', essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_nn_dropout2d_rng_state(mode, inplace):
    """
    Feature: mint.nn.Dropout2d
    Description: test function with random status.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    net = Dropout2d_nn(0.3, inplace)

    state = ms.get_rng_state()
    x = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out1 = net(x)

    y = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out2 = net(y)

    ms.set_rng_state(state)
    z = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out3 = net(z)

    if inplace:
        assert not (x.asnumpy() == y.asnumpy()).all()
        assert (x.asnumpy() == z.asnumpy()).all()
    else:
        assert not (out1.asnumpy() == out2.asnumpy()).all()
        assert (out1.asnumpy() == out3.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.0)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_func_dropout2d_generalize(mode):
    """
    Feature: mint.nn.functional.dropout
    Description: test function.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    for training in [True, False]:
        for p in [-3, -0.3]:
            with pytest.raises(ValueError):
                x = ms.Tensor(generate_ones_input((10, 10), np.float32))
                _ = dropout2d_forward_func(x, p, training, False)
                if mode == 'pynative':
                    _pynative_executor.sync()

    for inplace in [True, False]:
        for p in [True, 1, 1.0]:
            x = ms.Tensor(generate_ones_input((10, 10), np.float32))
            out = dropout2d_forward_func(x, p, True, inplace)
            expect = np.zeros_like(x.asnumpy())
            assert (out.asnumpy() == expect).all()
            if inplace:
                assert (x.asnumpy() == expect).all()

    for inplace in [True, False]:
        for p in [0.0, 0, False]:
            x = ms.Tensor(generate_ones_input((10, 10), np.float32))
            out = dropout2d_forward_func(x, p, True, inplace)
            assert (out.asnumpy() == x.asnumpy()).all()

        x = ms.Tensor(generate_ones_input((10, 10), np.float32))
        out = dropout2d_forward_func(x, 0.5, False, inplace)
        assert (out.asnumpy() == x.asnumpy()).all()

        x = ms.mint.empty((10, 0, 10), device="Ascend")
        out = dropout2d_forward_func(x, 0.5, True, inplace)
        assert (out.asnumpy() == x.asnumpy()).all()

    for shape in [(), (10,)]:
        x = ms.Tensor(generate_ones_input(shape, np.float32))
        p = 0.3
        with pytest.raises(ValueError):
            _ = dropout2d_forward_func(x, p, True, False)
            if mode == 'pynative':
                _pynative_executor.sync()

        out = dropout2d_forward_func(x, p, False, False)
        assert (out.asnumpy() == x.asnumpy()).all()

        for p in [0, 1]:
            _ = dropout2d_forward_func(x, p, False, False)
            if mode == 'pynative':
                _pynative_executor.sync()
