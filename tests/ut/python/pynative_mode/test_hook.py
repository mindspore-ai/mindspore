import numpy as np
import mindspore.nn as nn
import mindspore.ops.operations as P
from mindspore import context
from mindspore.ops import composite as C
from mindspore.common import dtype as mstype
from mindspore import context, Tensor, ParameterTuple
from mindspore.common.initializer import TruncatedNormal
from mindspore.nn import Dense, WithLossCell, SoftmaxCrossEntropyWithLogits, Momentum

context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0):
    """weight initial for conv layer"""
    weight = weight_variable()
    return nn.Conv2d(in_channels, out_channels,
                     kernel_size=kernel_size, stride=stride, padding=padding,
                     weight_init=weight, has_bias=False, pad_mode="valid")

def fc_with_initialize(input_channels, out_channels):
    """weight initial for fc layer"""
    weight = weight_variable()
    bias = weight_variable()
    return nn.Dense(input_channels, out_channels, weight, bias)

def weight_variable():
    """weight initial"""
    return TruncatedNormal(0.02)

def cell_hook_function(cell_id, grad_input, grad_output):
    print(cell_id)
    assert(grad_output[0].asnumpy().shape == (32, 6, 14, 14))
    assert(grad_input[0].asnumpy().shape == (32, 16, 10, 10))


def var_hook_function(grad_out):
    print("grad:", grad_out)
    assert(grad_out[0].asnumpy().shape == (32, 120))


class LeNet5(nn.Cell):
    """
    Lenet network
    Args:
        num_class (int): Num classes. Default: 10.
        
    Returns:
        Tensor, output tensor

    Examples:
        >>> LeNet(num_class=10)
    """
    def __init__(self, num_class=10):
        super(LeNet5, self).__init__()
        self.num_class = num_class
        self.batch_size = 32
        self.conv1 = conv(1, 6, 5)
        self.conv2 = conv(6, 16, 5)
        self.conv2.register_backward_hook(cell_hook_function)
        self.fc1 = fc_with_initialize(16 * 5 * 5, 120)
        self.fc2 = fc_with_initialize(120, 84)
        self.fc3 = fc_with_initialize(84, self.num_class)
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.reshape = P.Reshape()
        self.hook = P.HookBackward(var_hook_function)

    def construct(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.reshape(x, (self.batch_size, -1))
        x = self.fc1(x)
        x = self.hook(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

    
class GradWrap(nn.Cell):
    """ GradWrap definition """
    def __init__(self, network):
        super(GradWrap, self).__init__(auto_prefix=False)
        self.network = network
        self.weights = ParameterTuple(filter(lambda x: x.requires_grad, network.get_parameters()))

    def construct(self, x, label):
        weights = self.weights
        return C.GradOperation('get_by_list', get_by_list=True)(self.network, weights)(x, label)

def test_hook():
    net = LeNet5()
    optimizer = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.1, 0.9)
    criterion = nn.SoftmaxCrossEntropyWithLogits(is_grad=False, sparse=False)
    net_with_criterion = WithLossCell(net, criterion)
    train_network = GradWrap(net_with_criterion)
    train_network.set_train()
    
    input_data = Tensor(np.ones([net.batch_size, 1, 32, 32]).astype(np.float32) * 0.01)
    label = Tensor(np.ones([net.batch_size, net.num_class]).astype(np.float32))
    output = net(Tensor(input_data))
    loss_output = criterion(output, label)
    grads = train_network(input_data, label)
    success = optimizer(grads)
    print(loss_output.asnumpy().shape)




class MulAdd(nn.Cell):
    def __init__(self):
        super(MulAdd, self).__init__()

    def construct(self, x, y):
        return 2 * x + y

    def bprop(self, x, y, out, dout):
        assert(x == 1)
        assert(y == 2)
        assert(out == 4)
        assert(dout == 1)
        return 3 * dout, 2 * y

def test_custom_bprop():
    mul_add = MulAdd()
    mul_add.bprop_debug = True
    assert C.grad_all(mul_add)(1, 2) == (3, 4)
