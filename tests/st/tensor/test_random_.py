import numpy as np
import pytest
import mindspore as ms
from mindspore import ops, context
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


class Net(ms.nn.Cell):
    def construct(self, input_x, from_, to, generator=None):
        input_x = input_x + 1
        input_x.random_(from_, to, generator=generator)
        return input_x


def random_forward_func(input_x, from_, to, generator=None):
    return Net()(input_x, from_, to, generator=generator)


@test_utils.run_with_cell
def random_backward_func(input_x, from_, to, generator=None):
    return ops.grad(random_forward_func, (0))(input_x, from_, to, generator=generator)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', ['pynative'])
def test_random_normal(mode):
    """
    Feature: random_ function.
    Description: test function random_
    Expectation: expect correct result.
    """
    shape = (5, 5)
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    from_ = 0
    to = 1
    generator = None
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = random_forward_func(
            input_x, from_, to, generator=generator).asnumpy()
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output = random_forward_func(
            input_x, from_, to, generator=generator).asnumpy()
    assert np.all(output < 1) & np.all(output >= 0)
    assert output.dtype == np.float64

    # random_ backward
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = random_backward_func(
            input_x, from_, to, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        input_grad = random_backward_func(
            input_x, from_, to, generator=generator)
    assert np.all(input_grad.asnumpy() == 0)
    assert input_grad.asnumpy().dtype == np.float64


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential', mem_peak=1024.0)
@pytest.mark.parametrize('mode', ['pynative'])
def test_random_randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    shape = (5, 5)
    # Tensor.random_ is an inplace ops, so need to create two same tensor for testing
    np.random.seed(10)
    input_x1 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    np.random.seed(10)
    input_x2 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    from_ = 0
    to = 10000
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = random_forward_func(
            input_x1, from_, to, generator=generator)
        output2 = random_forward_func(
            input_x2, from_, to, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = random_forward_func(
            input_x1, from_, to, generator=generator)
        output2 = random_forward_func(
            input_x2, from_, to, generator=generator)
    assert np.any(output1.asnumpy() != output2.asnumpy())

    state = generator.get_state()
    np.random.seed(20)
    input_x1 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    np.random.seed(20)
    input_x2 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = random_forward_func(
            input_x1, from_, to, generator=generator)
        generator.set_state(state)
        output2 = random_forward_func(
            input_x2, from_, to, generator=generator)
    else:
        context.set_context(mode=ms.GRAPH_MODE)
        output1 = random_forward_func(
            input_x1, from_, to, generator=generator)
        generator.set_state(state)
        output2 = random_forward_func(
            input_x2, from_, to, generator=generator)
    assert np.all(output1.asnumpy() == output2.asnumpy())
