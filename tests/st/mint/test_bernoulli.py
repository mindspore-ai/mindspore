import numpy as np
import pytest
import mindspore as ms
from mindspore import mint, ops, context, jit
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


@test_utils.run_with_cell
def bernoulli_forward_func(input_x, generator=None):
    return mint.bernoulli(input_x, generator=generator)


@test_utils.run_with_cell
def bernoulli_backward_func(input_x, generator=None):
    return ops.grad(bernoulli_forward_func, (0))(input_x, generator)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_bernoulli_normal(mode):
    """
    Feature: bernoulli function.
    Description: test function bernoulli
    Expectation: expect correct result.
    """
    shape = (5, 5)
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    generator = None
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = bernoulli_forward_func(input_x, generator=generator)
    elif mode == 'kbk':
        context.set_context(mode=ms.GRAPH_MODE)
        output = (jit(bernoulli_forward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = bernoulli_forward_func(input_x, generator=generator)
    assert np.isin(output.asnumpy(), [0, 1]).all()
    assert output.asnumpy().dtype == np.float64

    # bernoulli backward
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = bernoulli_backward_func(input_x, generator=generator)
    elif mode == 'kbk':
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = (jit(bernoulli_backward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = bernoulli_backward_func(input_x, generator=generator)
    assert np.all(input_grad.asnumpy() == 0)
    assert input_grad.asnumpy().dtype == np.float64


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential', mem_peak=2048.01)
@pytest.mark.parametrize('mode', ['pynative', 'kbk'])
def test_bernoulli_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    shape = (5, 5)
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = bernoulli_forward_func(input_x, generator=generator)
        output2 = bernoulli_forward_func(input_x, generator=generator)
    elif mode == 'kbk':
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = (jit(bernoulli_forward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
        output2 = (jit(bernoulli_forward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = bernoulli_forward_func(input_x, generator=generator)
        output2 = bernoulli_forward_func(input_x, generator=generator)
    assert np.any(output1.asnumpy() != output2.asnumpy())

    state = generator.get_state()
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = bernoulli_forward_func(input_x, generator=generator)
        generator.set_state(state)
        output2 = bernoulli_forward_func(input_x, generator=generator)
    elif mode == 'kbk':
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = (jit(bernoulli_forward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
        generator.set_state(state)
        output2 = (jit(bernoulli_forward_func, backend="ms_backend", jit_level="O0"))(input_x, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = bernoulli_forward_func(input_x, generator=generator)
        generator.set_state(state)
        output2 = bernoulli_forward_func(input_x, generator=generator)
    assert np.all(output1.asnumpy() == output2.asnumpy())
