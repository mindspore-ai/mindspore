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
import numpy as np
import pytest
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.common.initializer import initializer
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")


class Grad(nn.Cell):
    """Grad Net"""
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


class Net(nn.Cell):
    """BN Net"""
    def __init__(self):
        super(Net, self).__init__()
        self.bn = P.BatchNorm(is_training=True)
        self.scale = Parameter(initializer('ones', [64]), name='scale')
        self.b = Parameter(initializer('zeros', [64]), name='b')
        self.mean = Parameter(initializer('ones', [64]), name='mean')
        self.variance = Parameter(initializer('zeros', [64]), name='variance')

    def construct(self, x):
        return self.bn(x, self.scale, self.b, self.mean, self.variance)[0]


@pytest.mark.level0
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [1, 64, 112, 112], run in ascend.
    Expectation: print output y.
    """
    x = np.random.randn(1, 64, 112, 112).astype(np.float32)
    sens = np.random.randn(1, 64, 112, 112).astype(np.float32)
    dynamic_x = Tensor(shape=[1, 64, None, None], dtype=mindspore.float32)
    sens_x = Tensor(shape=[1, 64, None, None], dtype=mindspore.float32)
    net = Grad(Net())
    net.set_inputs(dynamic_x, sens_x)
    net.set_train(True)
    net(Tensor(x), Tensor(sens))
