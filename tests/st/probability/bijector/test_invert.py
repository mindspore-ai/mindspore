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
"""test cases for invert"""
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
    def __init__(self):
        super(Net, self).__init__()
        self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
        self.invert = msb.Invert(self.origin)

    def construct(self, x_):
        return self.invert.forward(x_), self.origin.inverse(x_)

def test_forward():
    forward = Net()
    x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    ans, ans2 = forward(Tensor(x, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - ans2.asnumpy()) < tol).all()

class Net1(nn.Cell):
    """
    Test class: backward pass of bijector.
    """
    def __init__(self):
        super(Net1, self).__init__()
        self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
        self.invert = msb.Invert(self.origin)

    def construct(self, x_):
        return self.invert.inverse(x_), self.origin.forward(x_)

def test_backward():
    backward = Net1()
    x = np.array([2.0, 3.0, 4.0, 5.0]).astype(np.float32)
    ans, ans2 = backward(Tensor(x, dtype=dtype.float32))
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - ans2.asnumpy()) < tol).all()

class Net2(nn.Cell):
    """
    Test class: Forward Jacobian.
    """
    def __init__(self):
        super(Net2, self).__init__()
        self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
        self.invert = msb.Invert(self.origin)

    def construct(self, x_):
        return self.invert.forward_log_jacobian(x_),\
               self.origin.inverse_log_jacobian(x_)

def test_forward_jacobian():
    forward_jacobian = Net2()
    x = Tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype.float32)
    ans, ans2 = forward_jacobian(x)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - ans2.asnumpy()) < tol).all()

class Net3(nn.Cell):
    """
    Test class: Backward Jacobian.
    """
    def __init__(self):
        super(Net3, self).__init__()
        self.origin = msb.ScalarAffine(scale=2.0, shift=1.0)
        self.invert = msb.Invert(self.origin)

    def construct(self, x_):
        return self.invert.inverse_log_jacobian(x_),\
               self.origin.forward_log_jacobian(x_)

def test_backward_jacobian():
    backward_jacobian = Net3()
    x = Tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype.float32)
    ans, ans2 = backward_jacobian(x)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - ans2.asnumpy()) < tol).all()
