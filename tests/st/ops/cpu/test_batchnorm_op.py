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

import numpy as np
import pytest

import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import BatchNorm2d
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F


class BatchNormNet(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init):
        super(BatchNormNet, self).__init__()
        self.bn = BatchNorm2d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init)

    def construct(self, input_data):
        x = self.bn(input_data)
        return x


class Grad(Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True)
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
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias), Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_train_backward():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)
    grad = np.array([[
        [[1, 2, 7, 1], [4, 2, 1, 3], [1, 6, 5, 2], [2, 4, 3, 2]],
        [[9, 4, 3, 5], [1, 3, 7, 6], [5, 7, 9, 9], [1, 4, 6, 8]]]]).astype(np.float32)
    expect_output = np.array([[[[-0.69126546, -0.32903028, 1.9651246, -0.88445705],
                                [0.6369296, -0.37732816, -0.93275493, -0.11168876],
                                [-0.7878612, 1.3614, 0.8542711, -0.52222186],
                                [-0.37732816, 0.5886317, -0.11168876, -0.28073236]],

                               [[1.6447213, -0.38968924, -1.0174079, -0.55067265],
                                [-2.4305856, -1.1751484, 0.86250514, 0.5502673],
                                [0.39576983, 0.5470243, 1.1715001, 1.6447213],
                                [-1.7996241, -0.7051701, 0.7080077, 0.5437813]]]]).astype(np.float32)

    weight = Tensor(np.ones(2).astype(np.float32))
    bias = Tensor(np.ones(2).astype(np.float32))
    moving_mean = Tensor(np.ones(2).astype(np.float32))
    moving_var_init = Tensor(np.ones(2).astype(np.float32))
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-6

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    bn_net = BatchNormNet(2, weight, bias, moving_mean, moving_var_init)
    bn_net.set_train()
    bn_grad = Grad(bn_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


def test_batch_norm_forward_functional(nptype):
    """
    Feature: test batch_norm forward for given input dtype.
    Description: test inputs for given input dtype.
    Expectation: the result match with expected result.
    """
    input_x = Tensor(np.ones([2, 2]).astype(nptype))
    running_mean = Tensor(np.ones([2]).astype(nptype))
    running_var = Tensor(np.ones([2]).astype(nptype))
    weight = Tensor(np.ones([2]).astype(nptype))
    bias = Tensor(np.ones([2]).astype(nptype))
    output = F.batch_norm(input_x, running_mean, running_var, weight, bias)
    expected = np.array([[1., 1.], [1., 1.]]).astype(nptype)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_batch_norm_forward_float32_functional():
    """
    Feature: test batch_norm forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    test_batch_norm_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    test_batch_norm_forward_functional(np.float32)


if __name__ == '__main__':
    test_batch_norm_forward_float32_functional()
