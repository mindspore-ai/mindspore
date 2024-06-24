from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.hsvtorgb = P.HSVToRGB()

    def construct(self, x):
        return self.hsvtorgb(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_float16():
    """
    Feature: None
    Description: basic test float16
    Expectation: just test
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([0.5, 0.5, 0.5]).astype(np.float16).reshape([1, 1, 1, 3])
    net = Net()
    output = net(Tensor(x))
    expected = np.array([0.25, 0.5, 0.5]).astype(np.float16).reshape([1, 1, 1, 3])
    assert np.allclose(output.asnumpy(), expected, 1e-3, 1e-3)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_float32():
    """
    Feature: None
    Description: basic test float32
    Expectation: just test
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([0.5, 0.5, 0.5]).astype(np.float32).reshape([1, 1, 1, 3])
    net = Net()
    output = net(Tensor(x))
    expected = np.array([0.25, 0.5, 0.5]).astype(np.float32).reshape([1, 1, 1, 3])
    assert np.allclose(output.asnumpy(), expected, 1e-4, 1e-4)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_float64():
    """
    Feature: None
    Description: basic test float64
    Expectation: just test
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([0.5, 0.5, 0.5]).astype(np.float64).reshape([1, 1, 1, 3])
    net = Net()
    output = net(Tensor(x))
    expected = np.array([0.25, 0.5, 0.5]).astype(np.float64).reshape([1, 1, 1, 3])
    assert np.allclose(output.asnumpy(), expected, 1e-5, 1e-5)
