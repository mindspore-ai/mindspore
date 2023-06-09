import numpy as np
from mindspore import context, nn, ops, Tensor
import mindspore as ms

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Net(nn.Cell):

    def __init__(self):
        super(Net).__init__()

    def construct(self, input_params, input_indices, axis):
        out = ops.gather(input_params, input_indices, axis)
        return out


def test_aicore_error():
    """
    Feature: Print error message
    Description: Test log when an aicore error occurs
    Expectation: print log like "[ERROR] Task DebugString, Tbetask..."
    """
    input_params = Tensor(np.random.uniform(0, 1, size=(64,)).astype("float32"))
    input_indices = Tensor(np.array([1000000, 101]), ms.int32)
    axis = 0
    net = Net()
    net(input_params, input_indices, axis)
