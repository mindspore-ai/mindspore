# Copyright 2019 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""test cases for gumbel_cdf"""
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
from mindspore import Tensor
from mindspore import dtype

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

class Net(nn.Cell):
    """
    Test class: forward pass of bijector.
    """
    def __init__(self, loc, scale):
        super(Net, self).__init__()
        self.bijector = msb.GumbelCDF(loc, scale)

    def construct(self, x_):
        return self.bijector.forward(x_)

def test_forward():
    loc = np.array([0.0])
    scale = np.array([[1.0], [2.0]])
    forward = Net(loc, scale)
    x = np.array([-2., -1., 0., 1., 2.]).astype(np.float32)
    ans = forward(Tensor(x, dtype=dtype.float32))
    tol = 1e-6
    expected = np.exp(-np.exp(-(x - loc)/scale))
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net1(nn.Cell):
    """
    Test class: backward pass of bijector.
    """
    def __init__(self, loc, scale):
        super(Net1, self).__init__()
        self.bijector = msb.GumbelCDF(loc, scale)

    def construct(self, x_):
        return self.bijector.inverse(x_)

def test_backward():
    loc = np.array([0.0])
    scale = np.array([[1.0], [2.0]])
    backward = Net1(loc, scale)
    x = np.array([0.1, 0.25, 0.5, 0.75, 0.9]).astype(np.float32)
    ans = backward(Tensor(x, dtype=dtype.float32))
    tol = 1e-6
    expected = loc - scale * np.log(-np.log(x))
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net2(nn.Cell):
    """
    Test class: Forward Jacobian.
    """
    def __init__(self, loc, scale):
        super(Net2, self).__init__()
        self.bijector = msb.GumbelCDF(loc, scale)

    def construct(self, x_):
        return self.bijector.forward_log_jacobian(x_)

def test_forward_jacobian():
    loc = np.array([0.0])
    scale = np.array([[1.0], [2.0]])
    forward_jacobian = Net2(loc, scale)
    x = np.array([-2., -1., 0., 1., 2.]).astype(np.float32)
    ans = forward_jacobian(Tensor(x))
    z = (x - loc) / scale
    expected = -z - np.exp(-z) - np.log(scale)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net3(nn.Cell):
    """
    Test class: Backward Jacobian.
    """
    def __init__(self, loc, scale):
        super(Net3, self).__init__()
        self.bijector = msb.GumbelCDF(loc, scale)

    def construct(self, x_):
        return self.bijector.inverse_log_jacobian(x_)

def test_backward_jacobian():
    loc = np.array([0.0])
    scale = np.array([[1.0], [2.0]])
    backward_jacobian = Net3(loc, scale)
    x = np.array([0.1, 0.2, 0.5, 0.75, 0.9]).astype(np.float32)
    ans = backward_jacobian(Tensor(x))
    expected = np.log(scale / (-x * np.log(x)))
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()
