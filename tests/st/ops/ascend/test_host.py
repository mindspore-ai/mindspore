import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(enable_graph_kernel=False, save_graphs=False, mode=context.PYNATIVE_MODE, device_target="Ascend")

class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.host = P.Shape()
        self.t1 = Tensor(np.random.randn(16).astype(np.int32))

    def construct(self):
        return self.host(self.t1)

def test_net():
    """
    Feature: test host kernel in pynative mode
    Description: get shape
    Expectation: success
    """
    net = Net()
    out1 = net()
    print(out1.asnumpy())
