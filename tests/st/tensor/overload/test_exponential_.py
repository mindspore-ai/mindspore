import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
from mindspore import context
from mindspore.graph.api import _pynative_executor

class Net(ms.nn.Cell):
    def construct(self, input_x, lambd=1, generator=None):
        input_x.exponential_(lambd, generator=generator)
        return input_x

def exponential__forward_func(input_x, lambd, generator=None):
    net = Net()
    return net(input_x, lambd, generator=generator)


@arg_mark(plat_marks=['platform_ascend910b'], level_mark='level1', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_exponential__randomness(mode):
    """
    Feature: rand function exponential_.
    Description: test randomness of exponential_
    Expectation: expect correct result.
    """

    if mode == ms.PYNATIVE_MODE:
        context.set_context(mode=ms.PYNATIVE_MODE)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")

    generator = ms.Generator()
    shape = (5, 5)
    lambd = 2.0
    ## same random seed should have same output
    np.random.seed(10)
    generator.manual_seed(100)
    state = generator.get_state()

    input_x1 = ms.Tensor(np.random.rand(*shape), dtype=ms.float16)
    generator.set_state(state)
    output1 = exponential__forward_func(input_x1, lambd, generator=generator)
    _pynative_executor.sync()
    np.random.seed(10)
    generator.set_state(state)
    input_x2 = ms.Tensor(np.random.rand(*shape), dtype=ms.float16)
    output2 = exponential__forward_func(input_x2, lambd, generator=generator)

    assert np.allclose(output1.asnumpy(), output2.asnumpy())
