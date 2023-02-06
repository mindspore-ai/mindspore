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

import mindspore as ms
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.parameter import Parameter
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation


class BatchNormNet(nn.Cell):
    def __init__(self, is_training, data_format):
        super().__init__()
        self.bn = P.BatchNorm(is_training, 1e-5, 0.1, data_format)

    def construct(self, input_x, scale, bias, mean, variance):
        x = self.bn(input_x, scale, bias, mean, variance)
        return x


class Grad(nn.Cell):
    def __init__(self, network, sens):
        super().__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens = sens

    def construct(self, input_x, scale, bias, mean, variance):
        gout = self.grad(self.network)(input_x, scale, bias, mean, variance, self.sens)
        return gout


@pytest.mark.level1
@pytest.mark.platform_arm_ascend_training
@pytest.mark.platform_x86_ascend_training
@pytest.mark.env_onecard
def test_net():
    """
    Feature: Test acl call with pynative mode and dynamic shape.
    Description: Input Tensor with [2, 3, 16, 16], run in ascend.
    Expectation: run success.
    """
    channel = 3
    is_training = True
    data_format = "NCHW"
    x = np.random.randn(2, channel, 16, 16).astype(np.float32)
    scale = np.random.randn(channel).astype(np.float32)
    bias = np.random.randn(channel).astype(np.float32)
    mean = np.random.randn(channel).astype(np.float32)
    variance = np.random.randn(channel).astype(np.float32)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    bn_net = BatchNormNet(is_training, data_format)
    bn_net.set_inputs(Tensor(shape=[None, channel, 16, 16], dtype=ms.float32),
                      Parameter(scale), Parameter(bias), Parameter(mean), Parameter(variance))
    bn_net.set_train(True)
    output = bn_net(Tensor(x), Parameter(scale), Parameter(bias),
                    Parameter(mean), Parameter(variance))
    bn_grad_net = Grad(bn_net, sens=output)
    bn_grad_net(Tensor(x), Parameter(scale), Parameter(bias),
                Parameter(mean), Parameter(variance))
