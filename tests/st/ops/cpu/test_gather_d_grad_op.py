# Copyright 2019 Huawei Technologies Co., Ltd
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
from mindspore.ops import operations as P
from mindspore.common.api import jit
from mindspore.ops.composite import GradOperation

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetGatherD(nn.Cell):
    def __init__(self, dim=1):
        super(NetGatherD, self).__init__()
        self.gatherd = P.GatherD()
        self.dim = int(dim)

    def construct(self, x, index):
        return self.gatherd(x, self.dim, index)


class NetGatherDGrad(nn.Cell):
    def __init__(self, network):
        super(NetGatherDGrad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    @jit
    def construct(self, inputx, index, output_grad):
        return self.grad(self.network)(inputx, index, output_grad)


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherd_grad_fp32():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.float32) * prop
    index = np.random.randint(0, 5, (5, 3, 5)).astype(np.int32)
    dim = 1

    gatherd = NetGatherD(dim)
    grad = NetGatherDGrad(gatherd)
    dout = np.random.randint(0, 5, index.shape).astype(np.float32) * prop
    output_grad = grad(Tensor(x), Tensor(index), Tensor(dout))
    if isinstance(output_grad, (tuple, list)):
        output_grad = output_grad[0]
    print(output_grad.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherd_grad_fp16():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.float16) * prop
    index = np.random.randint(0, 5, (3, 5, 5)).astype(np.int32)
    dim = 0

    gatherd = NetGatherD(dim)
    grad = NetGatherDGrad(gatherd)
    dout = np.random.randint(0, 5, index.shape).astype(np.float16) * prop
    output_grad = grad(Tensor(x), Tensor(index), Tensor(dout))
    if isinstance(output_grad, (tuple, list)):
        output_grad = output_grad[0]
    print(output_grad.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherd_grad_int32():
    prop = 100 if np.random.random() > 0.5 else -100
    x = np.random.randn(5, 5, 5).astype(np.int32) * prop
    index = np.random.randint(0, 5, (5, 5, 7)).astype(np.int64)
    dim = -1

    gatherd = NetGatherD(dim)
    grad = NetGatherDGrad(gatherd)
    dout = np.random.randint(0, 5, index.shape).astype(np.int32) * prop
    output_grad = grad(Tensor(x), Tensor(index), Tensor(dout))
    if isinstance(output_grad, (tuple, list)):
        output_grad = output_grad[0]
    print(output_grad.asnumpy())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherd_grad_checkresult():
    x = np.array([[[-146.76097, 119.84371], [91.22607, -166.12923]],
                  [[37.67479, -8.696029], [43.804962, -23.369316]]], np.float32)
    index = np.array([[[0, 1], [0, 0]], [[0, 0], [0, 1]]], np.int32)
    dim = 1

    gatherd = NetGatherD(dim)
    grad = NetGatherDGrad(gatherd)
    dout = np.array([[[-1.23, 119.84], [91.22607, -145.67]], [[37.67479, -8.696029], [100.89, -23.369316]]], np.float32)
    output = grad(Tensor(x), Tensor(index), Tensor(dout))

    if isinstance(output, (tuple, list)):
        output = output[0]
    expect = np.array([[[89.99606, -145.67], [0., 119.84]], [[138.56479, -8.696029], [0., -23.369316]]], np.float32)
    error = np.ones(shape=expect.shape) * 1.0e-6
    assert np.all(np.abs(output.asnumpy() - expect) < error)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_gatherd_grad_dynamic_shape():
    """
    Feature: dynamic shape support of GatherDGrad.
    Description: input Tensor with dynamic shape.
    Expectation: output shape coincide with expect_shape.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='CPU')
    x_dyn = Tensor(shape=[2, None], dtype=ms.float16)
    x = Tensor(np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]]), dtype=ms.float16)
    dim = 0
    index_dyn = Tensor(shape=[None, 5], dtype=ms.int64)
    index = Tensor(np.array([[0, 1, 1, 0, 0], [1, 0, 0, 1, 1]]), dtype=ms.int64)
    grad_dyn = Tensor(shape=[2, None], dtype=ms.float16)
    grad = Tensor(np.array([[0.9031, 0.0890, 0.2779, 0.3198, 0.5710],
                            [0.6949, 0.8439, 0.2003, 0.6868, 0.4437]]), dtype=ms.float16)
    except_shape = (2, 5)
    grad_net = NetGatherDGrad(NetGatherD(dim))
    grad_net.set_inputs(x_dyn, index_dyn, grad_dyn)
    output = grad_net(x, index, grad)
    assert output[0].asnumpy().shape == except_shape
