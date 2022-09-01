# Copyright 2019-2022 Huawei Technologies Co., Ltd
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
from mindspore.common.parameter import ParameterTuple
from mindspore.nn import BatchNorm2d, BatchNorm1d, SGD
from mindspore.nn import Cell
from mindspore.ops import composite as C
from mindspore.ops import functional as F


class BatchNormNet(Cell):
    def __init__(self, c, weight, bias, moving_mean, moving_var_init, use_batch_statistics=None):
        super(BatchNormNet, self).__init__()
        self.bn = BatchNorm2d(c, eps=0.00001, momentum=0.1, beta_init=bias, gamma_init=weight,
                              moving_mean_init=moving_mean, moving_var_init=moving_var_init,
                              use_batch_statistics=use_batch_statistics)

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


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
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

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias),
                          Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias),
                          Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias),
                          Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias),
                          Tensor(moving_mean), Tensor(moving_var_init))
    bn_net.set_train(False)
    output = bn_net(Tensor(x))


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
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

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, weight, bias, moving_mean, moving_var_init)
    bn_net.set_train()
    bn_grad = Grad(bn_net)
    output = bn_grad(Tensor(x), Tensor(grad))
    diff = output[0].asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_train_stats_false_forward():
    x = np.array([[
        [[1, 3, 3, 5], [2, 4, 6, 8], [3, 6, 7, 7], [4, 3, 8, 2]],
        [[5, 7, 6, 3], [3, 5, 6, 7], [9, 4, 2, 5], [7, 5, 8, 1]]]]).astype(np.float32)

    expect_output = np.array([[[[3.707105, 5.121315, 5.121315, 6.535525],
                                [4.41421, 5.8284197, 7.24263, 8.656839],
                                [5.121315, 7.24263, 7.9497347, 7.9497347],
                                [5.8284197, 5.121315, 8.656839, 4.41421]],

                               [[6.535525, 7.9497347, 7.24263, 5.121315],
                                [5.121315, 6.535525, 7.24263, 7.9497347],
                                [9.363945, 5.8284197, 4.41421, 6.535525],
                                [7.9497347, 6.535525, 8.656839, 3.707105]]]]).astype(np.float32)

    weight = np.ones(2).astype(np.float32)
    bias = np.ones(2).astype(np.float32) * 3
    moving_mean = np.zeros(2).astype(np.float32)
    moving_var_init = np.ones(2).astype(np.float32) * 2
    error = np.ones(shape=[1, 2, 4, 4]) * 1.0e-4
    use_batch_statistics = False

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                          Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNormNet(2, Tensor(weight), Tensor(bias), Tensor(moving_mean),
                          Tensor(moving_var_init), use_batch_statistics)
    bn_net.set_train()
    output = bn_net(Tensor(x))
    diff = output.asnumpy() - expect_output
    assert np.all(diff < error)
    assert np.all(-diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_infer_backward():
    expect_output = np.array([[[[-0.3224156, -0.3840524], [1.1337637, -1.0998858]],
                               [[-0.1724273, -0.877854], [0.0422135, 0.5828123]],
                               [[-1.1006137, 1.1447179], [0.9015862, 0.5024918]]]]).astype(np.float32)
    np.random.seed(1)
    x_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    input_grad_np = np.random.randn(1, 3, 2, 2).astype(np.float32)
    ms_input = Tensor(x_np)
    weight = Tensor(np.ones(3).astype(np.float32))
    bias = Tensor(np.zeros(3).astype(np.float32))
    moving_mean = Tensor(np.zeros(3).astype(np.float32))
    moving_var_init = Tensor(np.ones(3).astype(np.float32))
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    ms_net = BatchNormNet(3, weight, bias, moving_mean, moving_var_init)
    ms_net.set_train(False)
    ms_grad = Grad(ms_net)
    ms_out_grad_np = ms_grad(ms_input, Tensor(input_grad_np))
    assert np.allclose(ms_out_grad_np[0].asnumpy(), expect_output)


class BatchNorm1DNet(Cell):
    def __init__(self, affine=True, gamma_init='ones', beta_init='zeros', moving_mean_init='zeros',
                 moving_var_init='ones', use_batch_statistics=None):
        super(BatchNorm1DNet, self).__init__()
        self.bn1 = BatchNorm1d(2, eps=0.00001, momentum=0.1, affine=affine, gamma_init=gamma_init, beta_init=beta_init,
                               moving_mean_init=moving_mean_init, moving_var_init=moving_var_init,
                               use_batch_statistics=use_batch_statistics)

    def construct(self, x):
        x = self.bn1(x)
        return x


class GradByListNet(Cell):
    def __init__(self, network):
        super(GradByListNet, self).__init__()
        self.grad = C.GradOperation(get_all=True, sens_param=True, get_by_list=True)
        self.network = network
        self.params = ParameterTuple(network.trainable_params())

    def construct(self, x, dy):
        grad_op = self.grad(self.network, self.params)
        output = grad_op(x, dy)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_1d_train():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    bn_net = BatchNorm1DNet(use_batch_statistics=None)
    grad_net = GradByListNet(bn_net)
    optimizer = SGD(bn_net.trainable_params(), learning_rate=0.01, momentum=0.9)
    bn_net.set_train(True)

    x1 = np.array([[1.6243454, -0.6117564],
                   [-0.5281718, -1.0729686],
                   [0.86540765, -2.3015387],
                   [1.7448118, -0.7612069],
                   [0.3190391, -0.24937038]]).astype(np.float32)
    dy1 = np.array([[1.4621079, -2.0601406],
                    [-0.3224172, -0.38405436],
                    [1.1337694, -1.0998913],
                    [-0.1724282, -0.8778584],
                    [0.04221375, 0.58281523]]).astype(np.float32)
    x2 = np.array([[-0.19183555, -0.887629],
                   [-0.7471583, 1.6924546],
                   [0.05080776, -0.6369957],
                   [0.19091548, 2.1002553],
                   [0.12015896, 0.6172031]]).astype(np.float32)
    dy2 = np.array([[0.30017033, -0.35224986],
                    [-1.1425182, -0.34934273],
                    [-0.20889424, 0.5866232],
                    [0.8389834, 0.9311021],
                    [0.2855873, 0.8851412]]).astype(np.float32)
    x_train = [x1, x2]
    dy_train = [dy1, dy2]

    dx1 = np.array([[0.8120, -2.0371],
                    [-0.2202, 0.5837],
                    [0.8040, 0.1950],
                    [-1.1823, -0.2786],
                    [-0.2135, 1.5371]]).astype(np.float32)
    gamma1 = np.array([0.9821, 0.9873]).astype(np.float32)
    beta1 = np.array([-0.0214, 0.0384]).astype(np.float32)
    mean1 = np.array([0.7246, -0.8994]).astype(np.float32)
    variance1 = np.array([0.9036, 0.6559]).astype(np.float32)

    dx2 = np.array([[1.1955, -0.4247],
                    [-0.2425, -0.6789],
                    [-1.4563, 0.3237],
                    [0.8752, 0.3351],
                    [-0.3719, 0.4448]]).astype(np.float32)
    gamma2 = np.array([0.9370, 0.9687]).astype(np.float32)
    beta2 = np.array([-0.0415, 0.0559]).astype(np.float32)
    mean2 = np.array([-0.0314, 0.4294]).astype(np.float32)
    variance2 = np.array([0.2213, 1.6822]).astype(np.float32)

    exp_dx = [dx1, dx2]
    exp_gamma = [gamma1, gamma2]
    exp_beta = [beta1, beta2]
    exp_mean = [mean1, mean2]
    exp_variance = [variance1, variance2]

    for data in zip(x_train, dy_train, exp_dx, exp_gamma, exp_beta, exp_mean, exp_variance):
        output = grad_net(Tensor(data[0]), Tensor(data[1]))
        assert np.allclose(output[0][0].asnumpy(), data[2], atol=1.0e-4)
        optimizer(output[1])
        assert np.allclose(bn_net.bn1.gamma.asnumpy(), data[3], atol=1.0e-4)
        assert np.allclose(bn_net.bn1.beta.asnumpy(), data[4], atol=1.0e-4)
        assert np.allclose(bn_net.bn1.moving_mean.asnumpy(), data[5], atol=1.0e-4)
        assert np.allclose(bn_net.bn1.moving_variance.asnumpy(), data[6], atol=1.0e-4)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_1d_eval():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    gamma_init = Tensor(np.array([0.93700373, 0.96870345]).astype(np.float32))
    beta_init = Tensor(np.array([-0.04145495, 0.05593072]).astype(np.float32))
    mean_init = Tensor(np.array([-0.03142229, 0.4294087]).astype(np.float32))
    variance_init = Tensor(np.array([0.2212921, 1.6822311]).astype(np.float32))
    bn_net = BatchNorm1DNet(affine=False, gamma_init=gamma_init, beta_init=beta_init, moving_mean_init=mean_init,
                            moving_var_init=variance_init, use_batch_statistics=None)
    bn_net.set_train(False)

    x1 = np.array([[-1.1006192, 1.1447237],
                   [0.9015907, 0.50249434],
                   [0.90085596, -0.68372786],
                   [-0.12289023, -0.93576944],
                   [-0.26788807, 0.53035545]]).astype(np.float32)
    x2 = np.array([[-0.7543979, 1.2528682],
                   [0.5129298, -0.29809284],
                   [0.48851815, -0.07557172],
                   [1.1316293, 1.5198169],
                   [2.1855755, -1.3964963]]).astype(np.float32)
    x_test = [x1, x2]

    y1 = np.array([[-2.1711, 0.5902],
                   [1.8169, 0.1105],
                   [1.8155, -0.7754],
                   [-0.2236, -0.9637],
                   [-0.5125, 0.1313]]).astype(np.float32)
    y2 = np.array([[-1.4815, 0.6710],
                   [1.0428, -0.4874],
                   [0.9942, -0.3212],
                   [2.2751, 0.8703],
                   [4.3744, -1.3078]]).astype(np.float32)
    y_test = [y1, y2]

    for x, y in zip(x_test, y_test):
        y_pred = bn_net(Tensor(x))
        assert np.allclose(y_pred.asnumpy(), y, atol=1.0e-4)


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
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_batch_norm_forward_float32_functional():
    """
    Feature: test batch_norm forward.
    Description: test float32 inputs.
    Expectation: the result match with expected result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    test_batch_norm_forward_functional(np.float32)
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    test_batch_norm_forward_functional(np.float32)
