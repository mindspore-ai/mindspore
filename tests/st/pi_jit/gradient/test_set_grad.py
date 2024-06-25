import pytest
import numpy as np
import mindspore.nn as nn
import mindspore.common.dtype as mstype
from mindspore import Tensor, jit, context, Parameter
from mindspore.ops import composite as C
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Dense(1, 1)
        self.forward_time = 0
        self.step_time = 1

    def inc_time(self):
        print("1st=>cur_time:", self.forward_time, " step:", self.step_time)
        self.forward_time += self.step_time
        print("2nd=>cur_time:", self.forward_time, " step:", self.step_time)

    @jit(mode="PIJit", jit_config={"compile_by_trace": False})
    def construct(self, x):
        x = self.fc(x)
        self.inc_time()
        return x


class CustomLoss(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.network = network
        self.network.set_grad()
        self.weight = Parameter(Tensor(np.random.rand(3, 1)))

    def construct(self, target, inputs):
        loss = target-self.network(inputs)
        grad = C.GradOperation(get_by_list=True, sens_param=False)
        gradients = grad(self.network, self.weight)(inputs)
        return loss, gradients


@arg_mark(plat_marks=['cpu_linux'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('Net1', [Net])
@pytest.mark.parametrize('CustomLoss1', [CustomLoss])
@pytest.mark.parametrize('x', [Tensor(np.array([[1.0], [2.0], [3.0]]), mstype.float32)])
@pytest.mark.parametrize('y', [Tensor(np.array([[2.0], [4.0], [6.0]]), mstype.float32)])
def test_forward_time(Net1, CustomLoss1, x, y):
    """
    Feature: Forward Time Evaluation
    Description: Test the forward pass time when set_grad() is used in the loss function's init() method.
    Expectation: The forward pass time of the network should be 1.
    """
    context.set_context(mode=context.PYNATIVE_MODE)
    net = Net1()
    loss = CustomLoss1(net)
    loss(y, x)
    assert net.forward_time == 1
