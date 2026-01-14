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
from mindspore import mint, nn
from mindspore.graph.api import _pynative_executor
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils
from tests.st.ops.test_tools.test_op import TEST_OP


def generate_ones_input(shape, dtype):
    return np.ones(shape).astype(dtype)


@test_utils.run_with_cell
def dropout_forward_func(x, p=0.4, inplace=False):
    return mint.nn.functional.dropout(x, p, True, inplace)


@test_utils.run_with_cell
def dropout_forward_func_grad(x, p=0.4, inplace=False):
    x = x * 1
    return mint.nn.functional.dropout(x, p, True, inplace)


@test_utils.run_with_cell
def dropout_backward_func(x, p=0.4, inplace=False):
    return ms.grad(dropout_forward_func_grad, (0))(x, p, inplace)


class Dropout_nn(nn.Cell):
    def __init__(self, p=0.5, inplace=False):
        super().__init__()
        self.net = mint.nn.Dropout(p, inplace)
        self.net.set_train()

    def construct(self, x):
        return self.net(x)


@test_utils.run_with_cell
def dropout_forward_nn(x, p=0.4, inplace=False):
    net = Dropout_nn(p, inplace)
    return net(x)


@test_utils.run_with_cell
def dropout_forward_nn_grad(x, p=0.4, inplace=False):
    net = Dropout_nn(p, inplace)
    x = x * 1
    return net(x)


@test_utils.run_with_cell
def dropout_backward_nn(x, p=0.4, inplace=False):
    return ms.grad(dropout_forward_nn_grad, (0))(x, p, inplace)


def compare_output(x, p, output):
    # check output
    keep_prob = 1 - p
    if output.dtype == ms.bfloat16:
        output_np = output.astype(ms.float32).asnumpy()
    else:
        output_np = output.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(output_np)
    assert (elem_count * (keep_prob - 0.02)) < nonzero_count < (elem_count * (keep_prob + 0.02))

    expect_sum = np.array(nonzero_count / (1 - p), dtype=np.float64)
    output_sum = np.sum(output_np.astype(np.float64))

    if output.dtype == ms.float32:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-3)
    else:
        np.testing.assert_allclose(output_sum, expect_sum, rtol=1e-2)


def compare_grad(x, p, grad):
    # check grad
    keep_prob = 1 - p
    if grad.dtype == ms.bfloat16:
        grad_np = grad.astype(ms.float32).asnumpy()
    else:
        grad_np = grad.asnumpy()
    elem_count = x.size
    nonzero_count = np.count_nonzero(grad_np)
    assert (elem_count * (keep_prob - 0.02)) < nonzero_count < (elem_count * (keep_prob + 0.02))


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_func_dropout_ext_normal(mode, inplace):
    """
    Feature: mint.nn.functional.dropout
    Description: test function dropout normal.
    Expectation: expect correct result.
    """
    x_np = generate_ones_input((1280, 77, 77), np.float32)
    x = ms.Tensor(x_np)
    p = 0.1

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
        inplace = False

    output = dropout_forward_func(x, p, inplace)
    compare_output(x_np, p, output)
    if inplace:
        compare_output(x_np, p, x)

    x1_np = generate_ones_input((3, 4096, 1280), np.float32)
    x1 = ms.Tensor(x1_np)
    p1 = 0.1
    grad = dropout_backward_func(x1, p1, inplace)
    compare_grad(x1_np, p1, grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='essential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_nn_dropout_ext_normal(mode, inplace):
    """
    Feature: mint.nn.Dropout
    Description: test function dropout normal.
    Expectation: success
    """
    x_np = generate_ones_input((1280, 77, 77), np.float32)
    x = ms.Tensor(x_np)
    p = 0.1

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')
        inplace = False

    output = dropout_forward_nn(x, p, inplace)
    compare_output(x_np, p, output)
    if inplace:
        compare_output(x_np, p, x)

    x1_np = generate_ones_input((3, 4096, 1280), np.float32)
    x1 = ms.Tensor(x1_np)
    p1 = 0.1
    grad = dropout_backward_nn(x1, p1, inplace)
    compare_grad(x1_np, p1, grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_func_dropout_ext_bfloat16(mode):
    """
    Feature: mint.nn.functional.dropout
    Description: test function dropout normal.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    x = generate_ones_input((128, 128), np.float32)
    p = 0.4
    output = dropout_forward_func(ms.Tensor(x, dtype=ms.bfloat16), p)
    compare_output(x, p, output)

    x1 = generate_ones_input((256, 256), np.float32)
    p1 = 0.3
    grad = dropout_backward_func(ms.Tensor(x1, dtype=ms.bfloat16), p1)
    compare_grad(x1, p1, grad)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_nn_dropout_ext_bf16(mode):
    """
    Feature: mint.nn.Dropout
    Description: bf16
    Expectation: success
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    x = generate_ones_input((128, 128), np.float32)
    p = 0.4
    output = dropout_forward_nn(ms.Tensor(x, dtype=ms.bfloat16), p)
    compare_output(x, p, output)

    x1 = generate_ones_input((256, 256), np.float32)
    p1 = 0.1
    grad = dropout_backward_nn(ms.Tensor(x1, dtype=ms.bfloat16), p1)
    compare_grad(x1, p1, grad)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
def test_func_dropout_ext_dynamic():
    """
    Feature: mint.nn.functional.dropout
    Description: dynamic
    Expectation: success
    """
    state = ms.get_rng_state()
    @test_utils.run_with_cell
    def dropout_ext_func(x, p, inplace):
        ms.set_rng_state(state)
        y = x * 1
        return mint.nn.functional.dropout(y, p, inplace=inplace)

    x1 = ms.Tensor(generate_ones_input((2, 3, 4), np.float32))
    x2 = ms.Tensor(generate_ones_input((2, 3, 4, 5), np.float32))
    p1 = 0.3
    p2 = 0.7

    TEST_OP(dropout_ext_func,
            [[x1, p1, False], [x2, p2, False]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'disable_input_check': True})

    TEST_OP(dropout_ext_func,
            [[x1, p1, True], [x2, p2, True]],
            disable_mode=["GRAPH_MODE_GE"],
            case_config={'disable_input_check': True})


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_func_dropout_ext_rng_state(mode, inplace):
    """
    Feature: mint.nn.functional.dropout
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
    out1 = dropout_forward_func(x, p, inplace)

    y = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out2 = dropout_forward_func(y, p, inplace)

    ms.set_rng_state(state)
    z = ms.Tensor(generate_ones_input((10, 10), np.float32))
    out3 = dropout_forward_func(z, p, inplace)

    if inplace:
        assert not (x.asnumpy() == y.asnumpy()).all()
        assert (x.asnumpy() == z.asnumpy()).all()
    else:
        assert not (out1.asnumpy() == out2.asnumpy()).all()
        assert (out1.asnumpy() == out3.asnumpy()).all()


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
@pytest.mark.parametrize('inplace', [True, False])
def test_nn_dropout_ext_rng_state(mode, inplace):
    """
    Feature: mint.nn.Dropout
    Description: test function with random status.
    Expectation: expect correct result.
    """
    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    net = Dropout_nn(0.3, inplace)

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
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['kbk', 'pynative'])
def test_func_dropout_generalize(mode):
    """
    Feature: mint.nn.functional.dropout
    Description: test function.
    Expectation: expect correct result.
    """
    @test_utils.run_with_cell
    def dropout_func(x, p, training=True, inplace=False):
        return mint.nn.functional.dropout(x, p, training, inplace)

    if mode == 'pynative':
        ms.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.set_context(mode=ms.GRAPH_MODE, jit_level='O0')

    for training in [True, False]:
        for p in [-3, -0.3]:
            with pytest.raises(ValueError):
                x = ms.Tensor(generate_ones_input((10, 10), np.float32))
                _ = dropout_func(x, p, training)
                if mode == 'pynative':
                    _pynative_executor.sync()

    for inplace in [True, False]:
        for p in [True, 1, 1.0]:
            x = ms.Tensor(generate_ones_input((10, 10), np.float32))
            out = dropout_func(x, p, True, inplace)
            expect = np.zeros_like(x.asnumpy())
            assert (out.asnumpy() == expect).all()
            if inplace:
                assert (x.asnumpy() == expect).all()

    for inplace in [True, False]:
        for p in [0.0, 0, False]:
            x = ms.Tensor(generate_ones_input((10, 10), np.float32))
            out = dropout_func(x, p, True, inplace)
            assert (out.asnumpy() == x.asnumpy()).all()

        x = ms.Tensor(generate_ones_input((10, 10), np.float32))
        out = dropout_func(x, 0.5, False, inplace)
        assert (out.asnumpy() == x.asnumpy()).all()

        x = ms.mint.empty((10, 0, 10), device="Ascend")
        out = dropout_func(x, 0.5, True, inplace)
        assert (out.asnumpy() == x.asnumpy()).all()
