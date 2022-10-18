import numpy as np
import pytest
import mindspore.context as context
from mindspore import Tensor
from mindspore.common.api import jit
import mindspore.nn as nn
from mindspore.ops.operations.math_ops import NextAfter


class NextAfterNet(nn.Cell):
    def __init__(self):
        super(NextAfterNet, self).__init__()
        self.nextafter = NextAfter()

    @jit
    def construct(self, x, y):
        return self.nextafter(x, y)


def nextafter_graph(x1, x2):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net_msp = NextAfterNet()
    out_msp = net_msp(Tensor(x1), Tensor(x2))
    return out_msp


def nextafter_pynative(x1, x2):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net_msp = NextAfterNet()
    out_msp = net_msp(Tensor(x1), Tensor(x2))
    return out_msp


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nextafter_float64_graph():
    """
    Feature: ALL To ALL
    Description: test cases for nextafter
    Expectation: the result match to tensorflow
    """
    x = np.array([0.0]).astype(np.float64)
    y = np.array([0.1]).astype(np.float64)
    out_tf = np.array([5.e-324]).astype(np.float64)
    out_msp = nextafter_graph(x, y)
    assert out_msp.asnumpy() == out_tf


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nextafter_float32_graph():
    """
    Feature: ALL To ALL
    Description: test cases for nextafter
    Expectation: the result match to tensorflow
    """
    x = np.array([[0.0], [0.1]]).astype(np.float32)
    y = np.array([[0.1], [0.2]]).astype(np.float32)
    out_tf = np.array([[1.4012985e-45], [1.0000001e-01]]).astype(np.float32)
    out_msp = nextafter_graph(x, y)
    assert (out_msp.asnumpy() == out_tf).all()


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nextafter_float64_pynative():
    """
    Feature: ALL To ALL
    Description: test cases for nextafter
    Expectation: the result match to tensorflow
    """
    x = np.array([0.0]).astype(np.float64)
    y = np.array([0.1]).astype(np.float64)
    out_tf = np.array([5.e-324]).astype(np.float64)
    out_msp = nextafter_pynative(x, y)
    assert out_msp.asnumpy() == out_tf


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_nextafter_float32_pynative():
    """
    Feature: ALL To ALL
    Description: test cases for nextafter
    Expectation: the result match to tensorflow
    """
    x = np.array([[0.0], [0.1]]).astype(np.float32)
    y = np.array([[0.1], [0.2]]).astype(np.float32)
    out_tf = np.array([[1.4012985e-45], [1.0000001e-01]]).astype(np.float32)
    out_msp = nextafter_pynative(x, y)
    assert (out_msp.asnumpy() == out_tf).all()
