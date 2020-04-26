import numpy as np

import mindspore
from mindspore import nn
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P


context.set_context(mode=context.GRAPH_MODE)


class Layer1(nn.Cell):
    def __init__(self):
        super(Layer1, self).__init__()
        self.net = nn.Conv2d(3, 1, 3, pad_mode='same')
        self.pad = nn.Pad(
            paddings=((0, 0), (0, 2), (0, 0), (0, 0)), mode="CONSTANT")

    def construct(self, x):
        y = self.net(x)
        return self.pad(y)


class Layer2(nn.Cell):
    def __init__(self):
        super(Layer2, self).__init__()
        self.net = nn.Conv2d(3, 1, 7, pad_mode='same')
        self.pad = nn.Pad(
            paddings=((0, 0), (0, 2), (0, 0), (0, 0)), mode="CONSTANT")

    def construct(self, x):
        y = self.net(x)
        return self.pad(y)


class Layer3(nn.Cell):
    def __init__(self):
        super(Layer3, self).__init__()
        self.net = nn.Conv2d(3, 3, 3, pad_mode='same')

    def construct(self, x):
        return self.net(x)


class SwitchNet(nn.Cell):
    def __init__(self):
        super(SwitchNet, self).__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()
        self.layer3 = Layer3()
        self.layers = (self.layer1, self.layer2, self.layer3)
        self.fill = P.Fill()

    def construct(self, x, index):
        y = self.layers[index](x)
        return y


class MySwitchNet(nn.Cell):
    def __init__(self):
        super(MySwitchNet, self).__init__()
        self.layer1 = Layer1()
        self.layer2 = Layer2()
        self.layer3 = Layer3()
        self.layers = (self.layer1, self.layer2, self.layer3)
        self.fill = P.Fill()

    def construct(self, x, index):
        y = self.layers[0](x)
        for i in range(len(self.layers)):
            if i == index:
                y = self.layers[i](x)
        return y


def test_layer_switch():
    net = MySwitchNet()
    x = Tensor(np.ones((3, 3, 24, 24)), mindspore.float32)
    index = Tensor(0, dtype=mindspore.int32)
    y = net(x, index)
