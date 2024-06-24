# Copyright 2022 Huawei Technologies Co., Ltd
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
from tests.mark_utils import arg_mark
import pytest
import numpy as np
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, Parameter
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self, var):
        super(Net, self).__init__()
        self.var = Parameter(var, name="var")
        self.apply_proximal_gradient_descent = P.ApplyProximalGradientDescent()

    def construct(self, alpha, l1, l2, delta):
        return self.apply_proximal_gradient_descent(self.var, alpha, l1, l2, delta)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_apply_proximal_gradient_descent_float32():
    """
    Feature: ApplyProximalGradientDescent gpu kernel
    Description: test the ApplyProximalGradientDescent.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    var = Tensor(np.ones([2, 2]).astype(np.float32))
    net = Net(var)
    alpha = 0.001
    l1 = 0.1
    l2 = 0.1
    delta = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]).astype(np.float32))
    output = net(alpha, l1, l2, delta)
    expect = np.array([[0.99969995, 0.99969995], [0.99969995, 0.99969995]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    var = Tensor(np.ones([2, 2]).astype(np.float32))
    net = Net(var)
    alpha = 0.001
    l1 = 0.1
    l2 = 0.1
    delta = Tensor(np.array([[0.1, 0.1], [0.1, 0.1]]).astype(np.float32))
    output = net(alpha, l1, l2, delta)
    expect = np.array([[0.99969995, 0.99969995], [0.99969995, 0.99969995]], dtype=np.float32)
    np.testing.assert_almost_equal(output.asnumpy(), expect)
    