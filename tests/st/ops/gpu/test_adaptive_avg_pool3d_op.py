# Copyright 2021 Huawei Technologies Co., Ltd
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

import torch
from torch.nn.functional import adaptive_avg_pool3d

import numpy as np
import pytest

import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
import mindspore.ops.operations.nn_ops as P
from mindspore.ops.operations import _grad_ops as G
from mindspore.common.api import ms_function

context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')


class Net(nn.Cell):
    def __init__(self, output_size):
        super(Net, self).__init__()
        self.adaptive_avg_pool3d = P.AdaptiveAvgPool3D(output_size)

    @ms_function
    def construct(self, x):
        return self.adaptive_avg_pool3d(x)


class GradNet(nn.Cell):
    def __init__(self):
        super(GradNet, self).__init__()
        self.adaptive_avg_pool3d_grad = G.AdaptiveAvgPool3DGrad()

    @ms_function
    def construct(self, x, dy):
        return self.adaptive_avg_pool3d_grad(x, dy)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("shape", [(1, 32, 9, 9, 9), (3, 9, 5, 4)])
def test_net_normal_with_functional(mode, shape):
    '''
    Feature: Test adaptive_avg_pool3d functional interface
    Description: A randomly generated 5-dimensional matrix, Expected pooled output size
    Expectation: Successfully get output with expected output size
    '''
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(*shape).astype(np.float32))
    output_size = (3, 4, 5)
    output = ops.adaptive_avg_pool3d(x, output_size)
    expect_shape = shape[:-3] + output_size
    assert output.asnumpy().shape == expect_shape

    output_size = 3
    output = ops.adaptive_avg_pool3d(x, output_size)
    expect_shape = shape[:-3] + (output_size, output_size, output_size)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
@pytest.mark.parametrize("shape", [(1, 32, 9, 9, 9), (3, 9, 5, 4)])
def test_net_normal_with_nn(mode, shape):
    '''
    Feature: Test AdaptiveAvgPool3d nn interface
    Description: A randomly generated 5-dimensional matrix, Expected pooled output size
    Expectation: Successfully get output with expected output size
    '''
    context.set_context(mode=mode)
    x = Tensor(np.random.randn(*shape).astype(np.float32))
    output_size = (3, 4, 5)
    net = nn.AdaptiveAvgPool3d(output_size)
    output = net(x)
    expect_shape = shape[:-3] + output_size
    assert output.asnumpy().shape == expect_shape

    output_size = 3
    output = ops.adaptive_avg_pool3d(x, output_size)
    expect_shape = shape[:-3] + (output_size, output_size, output_size)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_normal():
    '''
    Feature: If AdaptiveAvgPool3D is normal
    Description: A randomly generated 5-dimensional matrix, Expected pooled output size
    Expectation: Successfully get output with expected output size
    '''
    x = np.random.randn(1, 32, 9, 9, 9)
    net = Net((3, 4, 5))
    output = net(Tensor(x, mindspore.float32))
    expect_shape = (1, 32, 3, 4, 5)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_net_graph_mode_fp64():
    '''
    Feature: If every value type of AdaptiveAvgPool3D and AdaptiveAvgPool3DGrad are normal
    Description: A 4-dimensional matrix with different types, Expected pooled output size
    Expectation: Successfully get output with expected output value
    '''
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([[[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],

                  [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]],

                  [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]],
                   [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]]])

    adaptive_avg_pool_3d = P.AdaptiveAvgPool3D((2, 2, 2))
    output_fp16 = adaptive_avg_pool_3d(Tensor(x, mindspore.float16))
    output_fp32 = adaptive_avg_pool_3d(Tensor(x, mindspore.float32))
    output_fp64 = adaptive_avg_pool_3d(Tensor(x, mindspore.float64))

    torchx_fp16 = torch.tensor(x, requires_grad=True, dtype=torch.half)
    output_torch_fp16 = adaptive_avg_pool3d(torchx_fp16, (2, 2, 2))
    torchx_fp32 = torch.tensor(x, requires_grad=True, dtype=torch.float)
    output_torch_fp32 = adaptive_avg_pool3d(torchx_fp32, (2, 2, 2))
    torchx_fp64 = torch.tensor(x, requires_grad=True, dtype=torch.double)
    output_torch_fp64 = adaptive_avg_pool3d(torchx_fp64, (2, 2, 2))

    expect_shape = (3, 2, 2, 2)
    expect_output = np.array([[[[3.0, 4.0], [6.0, 7.0]],
                               [[3.0, 4.0], [6.0, 7.0]]],

                              [[[3.0, 4.0], [6.0, 7.0]],
                               [[3.0, 4.0], [6.0, 7.0]]],

                              [[[3.0, 4.0], [6.0, 7.0]],
                               [[3.0, 4.0], [6.0, 7.0]]]])

    assert (output_fp16.asnumpy() == expect_output).all
    assert output_fp32.asnumpy().shape == expect_shape
    assert (output_fp32.asnumpy() == expect_output).all
    assert output_fp64.asnumpy().shape == expect_shape
    assert (output_fp64.asnumpy() == expect_output).all


    assert output_torch_fp16.detach().numpy().shape == expect_shape
    assert (output_fp16.asnumpy() - output_torch_fp16.detach().numpy() == 0).all
    assert output_torch_fp32.detach().numpy().shape == expect_shape
    assert (output_fp32.asnumpy() - output_torch_fp32.detach().numpy() == 0).all
    assert output_torch_fp64.detach().numpy().shape == expect_shape
    assert (output_fp64.asnumpy() - output_torch_fp64.detach().numpy() == 0).all

    expect_dx = np.array([[[[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]]],

                          [[[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]]],

                          [[[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]],
                           [[0.75, 1.75, 1.0], [2.25, 5.0, 2.75], [1.5, 3.25, 1.75]]]])
    grad_net = GradNet()

    dx_fp16 = grad_net(output_fp16, Tensor(np.array([3, 3, 3, 3])).astype(np.int32))
    dx_fp32 = grad_net(output_fp32, Tensor(np.array([3, 3, 3, 3])).astype(np.int32))
    dx_fp64 = grad_net(output_fp64, Tensor(np.array([3, 3, 3, 3])).astype(np.int32))

    output_torch_fp16.backward(torch.DoubleTensor([[[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]]]))
    dx_torch_fp16 = torchx_fp16.grad
    output_torch_fp32.backward(torch.DoubleTensor([[[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]]]))
    dx_torch_fp32 = torchx_fp32.grad
    output_torch_fp64.backward(torch.DoubleTensor([[[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]],

                                                   [[[1.0, 1.0], [1.0, 1.0]],
                                                    [[1.0, 1.0], [1.0, 1.0]]]]))
    dx_torch_fp64 = torchx_fp64.grad

    assert dx_fp16.asnumpy().shape == x.shape
    assert (dx_fp16.asnumpy() == expect_dx).all
    assert dx_fp32.asnumpy().shape == x.shape
    assert (dx_fp32.asnumpy() == expect_dx).all
    assert dx_fp64.asnumpy().shape == x.shape
    assert (dx_fp64.asnumpy() == expect_dx).all

    assert dx_torch_fp16.detach().numpy().shape == x.shape
    assert (dx_fp16.asnumpy() - dx_torch_fp16.detach().numpy() == 0).all
    assert dx_torch_fp32.detach().numpy().shape == x.shape
    assert (dx_fp32.asnumpy() - dx_torch_fp32.detach().numpy() == 0).all
    assert dx_torch_fp64.detach().numpy().shape == x.shape
    assert (dx_fp64.asnumpy() - dx_torch_fp64.detach().numpy() == 0).all
