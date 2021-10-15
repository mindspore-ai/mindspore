# Copyright 2020 Huawei Technologies Co., Ltd
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
import pytest
import mindspore.nn as nn
import mindspore.nn.probability.bijector as msb
from mindspore import Tensor
from mindspore import dtype
from mindspore import context

skip_flag = context.get_context("device_target") == "CPU"


def test_init():
    b = msb.PowerTransform()
    assert isinstance(b, msb.Bijector)
    b = msb.PowerTransform(1.)
    assert isinstance(b, msb.Bijector)


def test_type():
    with pytest.raises(TypeError):
        msb.PowerTransform(power='power')
    with pytest.raises(TypeError):
        msb.PowerTransform(name=0.1)


class Net(nn.Cell):
    """
    Test class: forward and inverse pass of bijector.
    """
    def __init__(self):
        super(Net, self).__init__()
        self.b1 = msb.PowerTransform(power=0.)
        self.b2 = msb.PowerTransform()

    def construct(self, x_):
        ans1 = self.b1.inverse(self.b1.forward(x_))
        ans2 = self.b2.inverse(self.b2.forward(x_))
        return ans1 - ans2


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test1():
    """
    Test forward and inverse pass of powertransform bijector.
    """
    net = Net()
    x = Tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype.float32)
    ans = net(x)
    assert isinstance(ans, Tensor)


class Jacobian(nn.Cell):
    """
    Test class: forward and inverse pass of bijector.
    """
    def __init__(self):
        super(Jacobian, self).__init__()
        self.b1 = msb.PowerTransform(power=0.)
        self.b2 = msb.PowerTransform()

    def construct(self, x_):
        ans1 = self.b1.forward_log_jacobian(x_)
        ans2 = self.b2.forward_log_jacobian(x_)
        ans3 = self.b1.inverse_log_jacobian(x_)
        ans4 = self.b2.inverse_log_jacobian(x_)
        return ans1 - ans2 + ans3 - ans4


@pytest.mark.skipif(skip_flag, reason="not support running in CPU")
def test2():
    """
    Test jacobians of powertransform bijector.
    """
    net = Jacobian()
    x = Tensor([2.0, 3.0, 4.0, 5.0], dtype=dtype.float32)
    ans = net(x)
    assert isinstance(ans, Tensor)
