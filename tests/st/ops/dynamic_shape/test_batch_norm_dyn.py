# Copyright 2020-2022 Huawei Technologies Co., Ltd
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

"""test BatchNorm forward and backward dynamic shape"""

import numpy as np
import pytest

from mindspore import context
from mindspore import Tensor
from mindspore import Parameter
from mindspore.ops import BatchNorm
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore import ops as P


class BatchNormNet(Cell):
    def __init__(self, is_training, data_format, indices):
        super().__init__()
        self.bn = BatchNorm(is_training, 1e-5, 0.1, data_format)
        self.unique = P.Unique()
        self.gather = P.Gather()
        self.indices = indices

    def construct(self, input_x, scale, bias, mean, variance):
        unique_indices, _ = self.unique(self.indices)
        input_x = self.gather(input_x, unique_indices, 0)
        x = self.bn(input_x, scale, bias, mean, variance)
        return x


class Grad(Cell):
    def __init__(self, network, sens):
        super().__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
        self.network = network
        self.sens = sens

    def construct(self, input_x, scale, bias, mean, variance):
        gout = self.grad(self.network)(input_x, scale, bias, mean, variance, self.sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batchnorm_training_nchw_dynamic_shape():
    """
    Feature: test batchnorm op in traning mode and nchw input data on cpu.
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    channel = 10
    is_training = True
    data_format = "NCHW"
    x = np.random.randn(5, channel).astype(np.float32)

    scale = np.random.randn(channel).astype(np.float32)
    bias = np.random.randn(channel).astype(np.float32)
    mean = np.random.randn(channel).astype(np.float32)
    variance = np.random.randn(channel).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_net = BatchNormNet(is_training, data_format, Tensor([0, 1, 2, 3, 4]))
    bn_net.set_train()
    output = bn_net(Parameter(x), Parameter(scale), Parameter(bias),
                    Parameter(mean), Parameter(variance))

    assert output[0].asnumpy().shape == x.shape
    assert output[1].asnumpy().shape == (channel,)
    assert output[2].asnumpy().shape == (channel,)
    assert output[3].asnumpy().shape == (channel,)
    assert output[4].asnumpy().shape == (channel,)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_grad_net = Grad(bn_net, sens=output)
    grad = bn_grad_net(Parameter(x), Parameter(scale), Parameter(bias),
                       Parameter(mean), Parameter(variance))
    assert grad[0].asnumpy().shape == x.shape
    assert grad[1].asnumpy().shape == (channel,)
    assert grad[2].asnumpy().shape == (channel,)
    assert grad[3].asnumpy().shape == (channel,)
    assert grad[4].asnumpy().shape == (channel,)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batchnorm_not_training_nhwc_dynamic_shape():
    """
    Feature: test batchnorm op not in traning mode and nhcw input data on gpu.
    Description: test the ops in dynamic shape
    Expectation: expect correct shape result.
    """
    channel = 10
    is_training = False
    data_format = "NHWC"
    x = np.random.randn(5, channel).astype(np.float32)

    scale = np.random.randn(channel).astype(np.float32)
    bias = np.random.randn(channel).astype(np.float32)
    mean = np.random.randn(channel).astype(np.float32)
    variance = np.random.randn(channel).astype(np.float32)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNormNet(is_training, data_format, Tensor([0, 1, 2, 3, 4]))
    output = bn_net(Tensor(x), Tensor(scale), Tensor(bias), Tensor(mean), Tensor(variance))

    assert output[0].asnumpy().shape == x.shape
    assert output[1].asnumpy().shape == (channel,)
    assert output[2].asnumpy().shape == (channel,)
    assert output[3].asnumpy().shape == (channel,)
    assert output[4].asnumpy().shape == (channel,)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_grad_net = Grad(bn_net, sens=output)
    grad = bn_grad_net(Tensor(x), Tensor(scale), Tensor(bias), Tensor(mean), Tensor(variance))
    assert grad[0].asnumpy().shape == x.shape
    assert grad[1].asnumpy().shape == (channel,)
    assert grad[2].asnumpy().shape == (channel,)
    assert grad[3].asnumpy().shape == (channel,)
    assert grad[4].asnumpy().shape == (channel,)
