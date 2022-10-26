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
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops.composite import GradOperation
import mindspore as ms
import torch as t
from torch.autograd import Variable

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")


class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @ms_function
    def construct(self, input_, output_grad):
        return self.grad(self.network)(input_, output_grad)


def test_grad_3():
    """
    Feature: test bn_infer grad
    Description: test bn_infer grad with input tensor's type float32 and num_features=3
    Expectation: none.
    """
    sens = np.random.randn(1, 3, 2, 2).astype(np.float32)
    x = np.random.randn(1, 3, 2, 2).astype(np.float32)
    bn = nn.BatchNorm2d(num_features=3, use_batch_statistics=False)
    net = Grad(bn)
    input_dyn = Tensor(shape=[None, 3, None, None], dtype=ms.float32)
    sens_dyn = Tensor(shape=[None, 3, None, None], dtype=ms.float32)
    net.set_inputs(input_dyn, sens_dyn)
    ms_output = net(Tensor(x), Tensor(sens))

    torchnet = t.nn.BatchNorm2d(3, affine=True, track_running_stats=True)
    torchnet.eval()
    input_torch = Variable(t.tensor(x), requires_grad=True)
    outtorch = torchnet(input_torch)
    outtorch.backward(t.tensor(sens))

    assert np.allclose(ms_output[0].asnumpy(), input_torch.grad.numpy(), 0.0001, 0.0001)


def test_grad_64():
    """
    Feature: test bn_infer grad
    Description: test bn_infer grad with input tensor's type float32 and num_features=64
    Expectation: none.
    """
    sens = np.random.randn(1, 64, 112, 112).astype(np.float32)
    x = np.random.randn(1, 64, 112, 112).astype(np.float32)
    bn = nn.BatchNorm2d(num_features=64, use_batch_statistics=False)
    net = Grad(bn)
    input_dyn = Tensor(shape=[None, 64, None, None], dtype=ms.float32)
    sens_dyn = Tensor(shape=[None, 64, None, None], dtype=ms.float32)
    net.set_inputs(input_dyn, sens_dyn)
    ms_output = net(Tensor(x), Tensor(sens))

    torchnet = t.nn.BatchNorm2d(64, affine=True, track_running_stats=True)
    torchnet.eval()
    input_torch = Variable(t.tensor(x), requires_grad=True)
    outtorch = torchnet(input_torch)
    outtorch.backward(t.tensor(sens))

    assert np.allclose(ms_output[0].asnumpy(), input_torch.grad.numpy(), 0.0001, 0.0001)
