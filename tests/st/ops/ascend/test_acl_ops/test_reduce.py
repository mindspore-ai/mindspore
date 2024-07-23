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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import jit
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Grad(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True)
        self.network = network

    @jit
    def construct(self, input_):
        return self.grad(self.network)(input_)


class Net(nn.Cell):
    """ReLU Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.relu = P.ReduceSum()

    def construct(self, x):
        return self.relu(x)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [3, 64, 112, 112], run in ascend.
    Expectation: print output y.
    """
    x = np.random.randn(3, 64, 128, 128).astype(np.float32)
    dynamic_x = Tensor(shape=[3, 64, None, None], dtype=mindspore.float32)
    net = Grad(Net())
    net.set_inputs(dynamic_x)
    net(Tensor(x))
