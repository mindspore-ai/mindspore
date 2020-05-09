import numpy as np
from mindspore import context
import mindspore as ms
import mindspore.nn as nn
from mindspore.ops import operations as P
from mindspore import Tensor
from mindspore.common.api import _executor
from tests.ut.python.ops.test_math_ops import VirtualLoss
from mindspore.parallel import set_algo_parameters
from mindspore.parallel._utils import _reset_op_id as reset_op_id
import re

class NetWithLoss(nn.Cell):
    def __init__(self, network):
        super(NetWithLoss, self).__init__()
        self.loss = VirtualLoss()
        self.network = network

    def construct(self, x):
        predict = self.network(x)
        return self.loss(predict)

class Blockcell(nn.Cell):
    def __init__(self):
        super(Blockcell, self).__init__()
        self.bn = nn.BatchNorm2d(64, momentum=0.9)

    def construct(self, x):
        out = self.bn(x)
        return out

def getBlock():
    return Blockcell()

def test_two_bn():
    class Net(nn.Cell):
        def __init__(self):
            super().__init__()
            self.block1 = getBlock()
            self.block2 = getBlock()
            self.relu = P.ReLU()
            self.add = P.TensorAdd()
            self.bias = Tensor(np.ones([64, 64]), dtype=ms.float32)

        def construct(self, x):
            out = self.block1(x)
            out = self.relu(out)
            out = self.add(out, self.bias)
            out = self.block2(out)
            return out

    net = NetWithLoss(Net())
    x = Tensor(np.ones([64, 64]), dtype=ms.float32)
    context.set_context(save_graphs=True)
    context.set_auto_parallel_context(device_num=8, global_rank=0)
    context.set_auto_parallel_context(parallel_mode="auto_parallel")
    net.set_auto_parallel()
    set_algo_parameters(elementwise_op_strategy_follow=True)
    reset_op_id()

    _executor.compile(net, x, phase='train')
    strategies = _executor._get_strategy(net)
    assert len(strategies) == 4

    for (k, v) in strategies.items():
        if re.search('BatchNorm-op', k) is not None:
            assert v == [[8, 1], [1], [1], [1], [1]]
        elif re.search('TensorAdd-op', k) is not None:
            assert v == [[8, 1], [8, 1]]
        elif re.search('ReLU-op', k) is not None:
            assert v == [[8, 1]]
