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
import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations._grad_ops import RandomGammaGrad


class RandomGammaGradNet(nn.Cell):
    def __init__(self):
        super(RandomGammaGradNet, self).__init__()
        self.random_gamma_grad = RandomGammaGrad()

    def construct(self, alpha, sample):
        return self.random_gamma_grad(alpha, sample)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_random_gamma_grad_graph():
    """
    Feature:  RandomGammaGrad
    Description: test case for RandomGammaGrad of float32
    Expectation: The result are as expected
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    alpha = Tensor(np.array([2.]).astype(np.float32))
    sample = Tensor(np.array([6., 0.5, 3., 2.]).astype(np.float32))
    net = RandomGammaGradNet()
    output = net(alpha, sample)
    out_expect = np.array([1.7880158, 0.49802664, 1.3217986, 1.0862083]).astype(np.float32)
    diff0 = abs(output.asnumpy() - out_expect)
    error0 = np.ones(shape=out_expect.shape) * 1.0e-4
    assert np.all(diff0 < error0)
    assert output.shape == out_expect.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_random_gamma_grad_pynative():
    """
    Feature:  RandomGammaGrad
    Description: test case for RandomGammaGrad of float64
    Expectation: The result are as expected
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    alpha = Tensor(np.array([2., 0.5, 6., 12.]).astype(np.float64))
    sample = Tensor(np.array([1., 2., 3., 4.]).astype(np.float64))
    net = RandomGammaGradNet()
    output = net(alpha, sample)
    out_expect = np.array([0.75077869, 2.51188938, 0.71225584, 0.55666]).astype(np.float64)
    diff0 = abs(output.asnumpy() - out_expect)
    error0 = np.ones(shape=out_expect.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output.shape == out_expect.shape
