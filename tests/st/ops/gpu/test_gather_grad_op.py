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
import mindspore.ops.operations._grad_ops as P
from mindspore import Tensor

class GatherDGradNet(nn.Cell):
    def __init__(self, dim=0):
        super(GatherDGradNet, self).__init__()
        self.gather_d_grad = P.GatherDGrad(dim)

    def construct(self, index, grad):
        return self.gather_d_grad(index, grad)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int32_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    net = GatherDGradNet(dim)
    output = net(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int64_fp32():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    net = GatherDGradNet(dim)
    output = net(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int32_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    net = GatherDGradNet(dim)
    output = net(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_graph_int64_fp16():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    net = GatherDGradNet(dim)
    output = net(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int32_fp32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    output = P.GatherDGrad(dim)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int64_fp32():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float32)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float32)
    output = P.GatherDGrad(dim)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int32_fp16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int32)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    output = P.GatherDGrad(dim)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)

@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_gather_grad_pynative_int64_fp16():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    dim = 0
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), ms.int64)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), ms.float16)
    expect = np.array([[0.9031, 0.8439, 0.2003, 0.3198, 0.5710],
                       [0.6949, 0.0890, 0.2779, 0.6868, 0.4437]], np.float16)
    output = P.GatherDGrad(dim)(index, grad)
    error = 1e-4
    diff = output.asnumpy() - expect
    assert np.all(diff < error)
