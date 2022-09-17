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
import mindspore.nn as nn
import mindspore as ms
import mindspore.ops.operations as P
import mindspore.ops.operations._grad_ops as G
from mindspore.ops.composite import GradOperation
from mindspore import Tensor

class GatherDNet(nn.Cell):
    def __init__(self, dim=0):
        super(GatherDNet, self).__init__()
        self.gather_d = P.GatherD()
        self.dim = dim

    def construct(self, x, index):
        return self.gather_d(x, self.dim, index)


class GatherDGradNet(nn.Cell):
    def __init__(self, net):
        super(GatherDGradNet, self).__init__()
        self.net = net
        self.grad = GradOperation(get_all=True, sens_param=True)(self.net)

    def construct(self, x, index, grad):
        return self.grad(x, index, grad)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int32_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), ms.float32)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    net = GatherDNet(dim)
    grad_net = GradOperation(get_all=True, sens_param=True)(net)
    output = grad_net(x, index, grad)
    error = 1e-4
    diff = output[0].asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int64_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), ms.float32)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    net = GatherDNet(dim)
    grad_net = GradOperation(get_all=True, sens_param=True)(net)
    output = grad_net(x, index, grad)
    error = 1e-4
    diff = output[0].asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int32_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), ms.float16)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    net = GatherDNet(dim)
    grad_net = GradOperation(get_all=True, sens_param=True)(net)
    output = grad_net(x, index, grad)
    error = 1e-4
    diff = output[0].asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int64_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), ms.float16)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    net = GatherDNet(dim)
    grad_net = GradOperation(get_all=True, sens_param=True)(net)
    output = grad_net(x, index, grad)
    error = 1e-4
    diff = output[0].asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int32_fp32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 5)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    output = G.GatherDGrad(dim, x_shape)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int64_fp32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 5)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    output = G.GatherDGrad(dim, x_shape)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int32_fp16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 5)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    output = G.GatherDGrad(dim, x_shape)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int64_fp16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_shape = (2, 5)
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    output = G.GatherDGrad(dim, x_shape)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gatherd_grad_dynamic_shape():
    """
    Feature: dynamic shape support of GatherDGrad.
    Description: input Tensor with dynamic shape.
    Expectation: output shape coincide with expect_shape.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x_dyn = Tensor(shape=[2, None], dtype=ms.float16)
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), dtype=ms.float16)
    dim = 0
    index_dyn = Tensor(shape=[None, 5], dtype=ms.int64)
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), dtype=ms.int64)
    grad_dyn = Tensor(shape=[2, None], dtype=ms.float16)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), dtype=ms.float16)
    except_shape = (2, 5)
    grad_net = GatherDGradNet(GatherDNet(dim))
    grad_net.set_inputs(x_dyn, index_dyn, grad_dyn)
    output = grad_net(x, index, grad)
    assert output[0].asnumpy().shape == except_shape
