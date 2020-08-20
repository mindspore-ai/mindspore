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

import numpy as np
import pytest

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import BatchNorm2d
from mindspore.nn import Cell
from mindspore.ops import composite as C


class Batchnorm_Net(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init):
        super(Batchnorm_Net, self).__init__()
        self.bn = BatchNorm2d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(name="get_all", get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_data, sens):
        gout = self.grad(self.network)(input_data, sens)
        return gout


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_train_forward():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.6059, 0.3118, 0.3118, 1.2294],
                                [-0.1471, 0.7706, 1.6882, 2.6059],
                                [0.3118, 1.6882, 2.1471, 2.1471],
                                [0.7706, 0.3118, 2.6059, -0.1471]],

                               [[0.9119, 1.8518, 1.3819, -0.0281],
                                [-0.0281, 0.9119, 1.3819, 1.8518],
                                [2.7918, 0.4419, -0.4981, 0.9119],
                                [1.8518, 0.9119, 2.3218, -0.9680]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32)
    moving_mean = np.ones(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32)
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_net = Batchnorm_Net(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))
