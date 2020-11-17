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
"""test cases for powertransform"""
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
    def __init__(self, power):
        super(Net, self).__init__()
        self.bijector = msb.PowerTransform(power=power)

    def construct(self, x_):
        forward = self.bijector.forward(x_)
        return forward

def test_forward():
    power = 2.
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    tx = Tensor(x, dtype=dtype.float32)
    forward = Net(power=power)
    ans = forward(tx)
    expected = np.exp(np.log1p(x * power) / power)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net1(nn.Cell):
    """
    Test class: inverse pass of bijector.
    """
    def __init__(self, power):
        super(Net1, self).__init__()
        self.bijector = msb.PowerTransform(power=power)

    def construct(self, y_):
        inverse = self.bijector.inverse(y_)
        return inverse

def test_inverse():
    power = 2.
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    ty = Tensor(y, dtype=dtype.float32)
    inverse = Net1(power=power)
    ans = inverse(ty)
    expected = np.expm1(np.log(y) * power) / power
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net2(nn.Cell):
    """
    Test class: Forward Jacobian.
    """
    def __init__(self, power):
        super(Net2, self).__init__()
        self.bijector = msb.PowerTransform(power=power)

    def construct(self, x_):
        return self.bijector.forward_log_jacobian(x_)

def test_forward_jacobian():
    power = 2.
    x = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    tx = Tensor(x, dtype=dtype.float32)
    forward_jacobian = Net2(power=power)
    ans = forward_jacobian(tx)
    expected = (1 / power - 1) * np.log1p(x * power)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()

class Net3(nn.Cell):
    """
    Test class: Backward Jacobian.
    """
    def __init__(self, power):
        super(Net3, self).__init__()
        self.bijector = msb.PowerTransform(power=power)

    def construct(self, y_):
        return self.bijector.inverse_log_jacobian(y_)

def test_inverse_jacobian():
    power = 2.
    y = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    ty = Tensor(y, dtype=dtype.float32)
    inverse_jacobian = Net3(power=power)
    ans = inverse_jacobian(ty)
    expected = (power - 1) * np.log(y)
    tol = 1e-6
    assert (np.abs(ans.asnumpy() - expected) < tol).all()
