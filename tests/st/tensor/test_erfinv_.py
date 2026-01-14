import numpy as np
import pytest
import mindspore as ms
from mindspore import Tensor, context, ops, mint
from tests.st.ops.test_tools.test_op import TEST_OP
from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


class Net(ms.nn.Cell):
    def construct(self, input_x):
        out = input_x.erfinv_()
        return out, input_x

class NetBackward(ms.nn.Cell):
    def construct(self, input_x):
        input_x = mint.add(input_x, 0)
        out = input_x.erfinv_()
        return out

def erfinv__forward_func(input_x):
    return Net()(input_x)

def erfinv__forward_backward_func(input_x):
    return NetBackward()(input_x)

@test_utils.run_with_cell
def erfinv__backward_func(input_x):
    return ops.grad(erfinv__forward_backward_func, (0))(input_x)

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=0.01)
@pytest.mark.parametrize('mode', ['KBK', 'PYNATIVE'])
def test_t_erfinv__normal(mode):
    """
    Feature: erfinv_ function.
    Description: test function erfinv_
    Expectation: expect correct result.
    """
    # Tensor.erfinv_ forward
    shape = (5, 5)
    input_x = ms.Tensor(np.ones(shape), dtype=ms.float32) * 0.5
    expect = np.array([[0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613],
                       [0.47693613, 0.47693613, 0.47693613, 0.47693613, 0.47693613]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        output, input_x = erfinv__forward_func(input_x)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        output, input_x = erfinv__forward_func(input_x)
    assert output.asnumpy().dtype == np.float32
    assert np.allclose(output.asnumpy(), expect)
    assert np.all(output.asnumpy() == input_x.asnumpy())

    # erfinv_ backward
    input_x = ms.Tensor(np.ones(shape), dtype=ms.float32) * 0.5
    expect = np.array([[1.11258471, 1.11258471, 1.11258471, 1.11258471, 1.11258471],
                       [1.11258471, 1.11258471, 1.11258471, 1.11258471, 1.11258471],
                       [1.11258471, 1.11258471, 1.11258471, 1.11258471, 1.11258471],
                       [1.11258471, 1.11258471, 1.11258471, 1.11258471, 1.11258471],
                       [1.11258471, 1.11258471, 1.11258471, 1.11258471, 1.11258471]])
    if mode == 'pynative':
        context.set_context(mode=ms.PYNATIVE_MODE)
        input_grad = erfinv__backward_func(input_x)
    else:
        ms.context.set_context(mode=ms.GRAPH_MODE, jit_level="O0")
        input_grad = erfinv__backward_func(input_x)
    assert np.allclose(input_grad.asnumpy(), expect)
    assert input_grad.asnumpy().dtype == np.float32

@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='essential', mem_peak=2048.0)
def test_t_erfinv__dynamic():
    """
    Feature: test dynamic erfinv_.
    Description: test dynamic of op InplaceErfinv.
    Expectation: expect correct result.
    """
    input_1 = Tensor(np.zeros((5, 5)), dtype=ms.float32)
    input_2 = Tensor(np.ones((3, 4, 5)), dtype=ms.float32)
    # dynamic string is not supported
    TEST_OP(erfinv__forward_backward_func, [[input_1], [input_2]],
            disable_mode=["GRAPH_MODE_GE"])
