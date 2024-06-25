import pytest
import numpy as np
import mindspore as ms
from mindspore import jit, Tensor, nn, ops, context
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sin = ops.Sin()
        self.cos = ops.Cos()

    @jit(mode="PIJit")
    def construct(self, x, y):
        out1 = self.sin(x) - self.cos(y)
        out2 = self.cos(x) - self.sin(y)
        return out1, out2


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('x_train', [Tensor(np.array([3.1415926]), dtype=ms.float32)])
@pytest.mark.parametrize('y_train', [Tensor(np.array([3.1415926]), dtype=ms.float32)])
@pytest.mark.parametrize('Net1', [Net])
def test_run(x_train, y_train, Net1):
    """
    Feature:
    Test the implementation of custom sin and cos operations in a MindSpore neural network.

    Description:
    1. Instantiate a custom network using MindSpore that utilizes sin and cos operations.
    2. Compute the first and second order gradients of the network using MindSpore's ops.grad function.
    3. Execute the network and acquire the output.

    Expectation:
    The second order gradients for both sin and cos should match the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net1()
    firstgrad = ops.grad(net, grad_position=(0, 1))
    secondgrad = ops.grad(firstgrad, grad_position=(0, 1))
    output = secondgrad(x_train, y_train)

    assert np.around(output[0].asnumpy(), decimals=2) == np.array([1])
    assert np.around(output[1].asnumpy(), decimals=2) == np.array([-1])
