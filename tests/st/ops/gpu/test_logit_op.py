import numpy as np
import pytest
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore import ops


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.logit = ops.logit

    def construct(self, x, eps):
        return self.logit(x, eps)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logit_graph():
    """
     Feature: Logit gpu TEST.
     Description: 1d test case for Logit with GRAPH_MODE
     Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
    eps = 1e-5
    net = Net()
    output = net(x, eps)
    expect = np.array([-2.19722462, -1.38629436, -0.84729779]).astype(np.float32)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * 1e-4
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_logit_pynative():
    """
     Feature: Logit gpu TEST.
     Description: 1d test case for Logit with PYNATIVE_MODE
     Expectation: The value and shape of output are the expected values.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.array([0.1, 0.2, 0.3]).astype(np.float32))
    eps = 1e-5
    logit = ops.logit
    output = logit(x, eps)
    expect = np.array([-2.19722462, -1.38629436, -0.84729779]).astype(np.float32)
    diff = output.asnumpy() - expect
    error = np.ones(shape=expect.shape) * 1e-4
    assert np.all(diff < error)
