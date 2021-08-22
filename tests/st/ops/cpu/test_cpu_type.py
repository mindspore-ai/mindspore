import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
import mindspore.context as context
from mindspore.nn import Dense
from mindspore.nn import TrainOneStepCell, WithLossCell
from mindspore.nn.optim import Momentum

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.bias_add = P.BiasAdd()
        self.bias_add1 = P.BiasAdd()

    def construct(self, x, b, c):
        return self.bias_add1(self.bias_add(x, b), c)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add1():
    x = np.ones([2, 2]).astype(np.float16)
    b = np.array([1, 1]).astype(np.float16)
    c = np.array([1, 1]).astype(np.float16)
    bias_add = Net()
    output = bias_add(Tensor(x), Tensor(b), Tensor(c))
    expect_output = np.ones([2, 2]).astype(np.float16) * 3
    assert np.all(output.asnumpy() == expect_output)


class Net1(nn.Cell):
    def __init__(self):
        super(Net1, self).__init__()
        self.bias_add = P.BiasAdd()
        self.mul = P.Mul()

    def construct(self, x, a, b):
        p1 = self.bias_add(x, b)
        p2 = self.bias_add(x, a)
        p3 = self.mul(p1, p2)
        return p3


class Net2(nn.Cell):
    def __init__(self):
        super(Net2, self).__init__()
        self.bias_add = P.BiasAdd()
        self.bias_add1 = P.BiasAdd()

    def construct(self, x, b, c):
        return self.bias_add1(self.bias_add(x, b), c)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_bias_add2():
    x = np.ones([2, 2]).astype(np.float32)
    a = np.array([1, 1]).astype(np.float32)
    b = np.array([1, 1]).astype(np.float32)
    c = np.array([1, 1]).astype(np.float32)
    bias_add = Net1()
    output = bias_add(Tensor(x), Tensor(a), Tensor(b))
    print(output)

    net2 = Net2()
    output2 = net2(Tensor(x), Tensor(b), Tensor(c))
    print(output2)


context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class MomentumNet(nn.Cell):
    def __init__(self):
        super(MomentumNet, self).__init__()
        self.batch_size = 1

        self.reshape = P.Reshape()
        weight = Tensor(np.ones([10, 16]).astype(np.float32) * 0.01)
        self.fc1 = Dense(16, 10, weight_init=weight)

    def construct(self, input_x):
        output = self.reshape(input_x, (self.batch_size, -1))
        output = self.fc1(output)
        return output


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_momentum():
    epoch = 1
    net = MomentumNet()
    learning_rate = (0.1, 0.2)
    momentum = 0.9

    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), learning_rate, momentum)
    criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    net_with_criterion = WithLossCell(net, criterion)
    train_network = TrainOneStepCell(net_with_criterion, optimizer)  # optimizer
    train_network.set_train()
    losses = []
    for _ in range(epoch):
        data = Tensor(np.arange(0, 16).reshape(1, 1, 4, 4).astype(np.float32) * 0.01)
        label = Tensor(np.array([0]).astype(np.int32))
        loss = train_network(data, label)
        losses.append(loss)
    print("================================")
    print(losses)

    return losses
