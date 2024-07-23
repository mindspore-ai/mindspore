from tests.mark_utils import arg_mark
import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P
import mindspore.common.dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.apply_ada_max = P.ApplyAdaMax()
        self.var = Parameter(Tensor(np.array([[0.6, 0.4],
                                              [0.1, 0.5]]).astype(np.float32)), name="var")
        self.m = Parameter(Tensor(np.array([[0.6, 0.5],
                                            [0.2, 0.6]]).astype(np.float32)), name="m")
        self.v = Parameter(Tensor(np.array([[0.9, 0.1],
                                            [0.7, 0.8]]).astype(np.float32)), name="v")

    def construct(self, beta1_power, lr, beta1, beta2, epsilon, grad):
        out = self.apply_ada_max(self.var, self.m, self.v, beta1_power, lr, beta1, beta2, epsilon, grad)
        return out


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_apply_ada_max():
    """
    Feature: ApplyAdaMax Operator on CPU
    Description: Test ApplyAdaMax Operator
    Expectation: Consistent with the results calculated using numpy
    """
    # ms
    net = Net()
    beta1_power = Tensor(0.9, mstype.float32)
    lr = Tensor(0.001, mstype.float32)
    beta1 = Tensor(0.9, mstype.float32)
    beta2 = Tensor(0.99, mstype.float32)
    epsilon = Tensor(1e-10, mstype.float32)
    grad = Tensor(np.array([[0.3, 0.7], [0.1, 0.8]]).astype(np.float32))
    output = net(beta1_power, lr, beta1, beta2, epsilon, grad)

    # numpy
    np_var = np.array([[0.6, 0.4], [0.1, 0.5]])
    np_m = np.array([[0.6, 0.5], [0.2, 0.6]])
    np_v = np.array([[0.9, 0.1], [0.7, 0.8]])
    np_beta1_power = 0.9
    np_lr = 0.001
    np_beta1 = 0.9
    np_beta2 = 0.99
    np_epsilon = 1e-10
    np_grad = np.array([[0.3, 0.7], [0.1, 0.8]])

    np_m = np_beta1 * np_m + (1.0 - np_beta1) * np_grad
    np_v = np.maximum(np_beta2 * np_v, abs(np_grad))
    np_var = np_var - (np_lr / (1 - np_beta1_power)) * (np_m / (np_v + np_epsilon))

    ms_m = output[1].asnumpy()
    ms_v = output[2].asnumpy()
    ms_var = output[0].asnumpy()
    eps = np.array([1e-6 for i in range(4)]).reshape(2, 2)
    assert np.all(np_m - ms_m < eps)
    assert np.all(np_v - ms_v < eps)
    assert np.all(np_var - ms_var < eps)
