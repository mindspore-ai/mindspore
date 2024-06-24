# Copyright 2023 Huawei Technologies Co., Ltd
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
from mindspore import Tensor, ops
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class Net(nn.Cell):
    def __init__(self, numsamples, replacement):
        super(Net, self).__init__()
        self.numsamples = numsamples
        self.replacement = replacement
        self.multinomialwithreplacement = P.random_ops.MultinomialWithReplacement(numsamples, replacement)

    def construct(self, x, seed, offset):
        return self.multinomialwithreplacement(x, seed, offset)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multinomial_with_replacement_net():
    """
    Feature: MultinomialWithReplacement cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    x = Tensor(np.random.randint(low=10, high=100, size=(10, 10)).astype(np.float32))
    seed = Tensor([50])
    offset = Tensor([20])
    net0 = Net(5, True)
    net1 = Net(10, True)
    net2 = Net(15, True)
    out0 = net0(x, seed, offset)
    out1 = net1(x, seed, offset)
    out2 = net2(x, seed, offset)
    assert out0.asnumpy().shape == (10, 5)
    assert out1.asnumpy().shape == (10, 10)
    assert out2.asnumpy().shape == (10, 15)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_multinomial_with_replacement_functional():
    """
    Feature: MultinomialWithReplacement cpu kernel
    Description: test the correctness of shape and result
    Expectation: success.
    """
    x = Tensor(np.random.randint(low=10, high=100, size=(10, 10)).astype(np.float32))
    seed = Tensor([50])
    offset = Tensor([20])
    output = ops.multinomial_with_replacement(x, seed, offset, 10, True)
    assert output.shape == (10, 10)
