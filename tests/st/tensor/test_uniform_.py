import numpy as np
import pytest
import mindspore as ms
from mindspore import mint
from mindspore import ops, context
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


class Net(ms.nn.Cell):
    def construct(self, input_x, from_, to, generator=None):
        input_x.uniform_(from_, to, generator=generator)
        return input_x

class NetBackward(ms.nn.Cell):
    def construct(self, input_x, from_, to):
        input_x = mint.add(input_x, 0)
        input_x.uniform_(from_, to)
        return input_x

def uniform__forward_func(input_x, from_, to, generator=None):
    return Net()(input_x, from_, to, generator=generator)

def uniform__forward_backward_func(input_x, from_, to):
    return NetBackward()(input_x, from_, to)

@test_utils.run_with_cell
def uniform__backward_func(input_x, from_, to):
    return ops.grad(uniform__forward_backward_func, (0))(input_x, from_, to)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.00)
@pytest.mark.parametrize('mode', ['KBK', 'PYNATIVE'])
def test_uniform__normal(mode):
    """
    Feature: uniform_ function.
    Description: test function uniform_
    Expectation: expect correct result.
    """
    shape = (5, 5)
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    from_ = 0
    to = 1
    generator = None
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output = uniform__forward_func(input_x, from_, to, generator=generator).asnumpy()
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output = uniform__forward_func(input_x, from_, to, generator=generator).asnumpy()
    assert np.all(output < 1) & np.all(output >= 0)
    assert output.dtype == np.float64

    # uniform_ backward
    input_x = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = uniform__backward_func(input_x, from_, to)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        input_grad = uniform__backward_func(input_x, from_, to)
    assert np.all(input_grad.asnumpy() == 0)
    assert input_grad.asnumpy().dtype == np.float64


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['KBK', 'PYNATIVE'])
def test_uniform__randomness(mode):
    """
    Feature: rand function.
    Description: test randomness of rand
    Expectation: expect correct result.
    """
    generator = ms.Generator()
    generator.seed()

    shape = (5, 5)
    # Tensor.uniform_ is an inplace ops, so need to create two same tensor for testing
    np.random.seed(10)
    input_x1 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    np.random.seed(10)
    input_x2 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    from_ = 0
    to = 1
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = uniform__forward_func(input_x1, from_, to, generator=generator)
        output2 = uniform__forward_func(input_x2, from_, to, generator=generator)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output1 = uniform__forward_func(input_x1, from_, to, generator=generator)
        output2 = uniform__forward_func(input_x2, from_, to, generator=generator)
    assert np.any(output1.asnumpy() != output2.asnumpy())

    state = generator.get_state()
    np.random.seed(20)
    input_x1 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    np.random.seed(20)
    input_x2 = ms.Tensor(np.random.rand(*shape), dtype=ms.float64)
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output1 = uniform__forward_func(input_x1, from_, to, generator=generator)
        generator.set_state(state)
        output2 = uniform__forward_func(input_x2, from_, to, generator=generator)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output1 = uniform__forward_func(input_x1, from_, to, generator=generator)
        generator.set_state(state)
        output2 = uniform__forward_func(input_x2, from_, to, generator=generator)
    assert np.all(output1.asnumpy() == output2.asnumpy())
