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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops.operations import _grad_ops as G


class PdistGradNet(nn.Cell):
    def __init__(self, p=2.0):
        super().__init__()
        self.pdistgrad = G.PdistGrad(p=p)

    def construct(self, y_grad, x, y):
        return self.pdistgrad(y_grad, x, y)


def pdist_grad_graph(y_grad, x, y, p):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = PdistGradNet(p)
    output_ms = net(y_grad, x, y)
    return output_ms


def pdist_grad_pynative(y_grad, x, y, p):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = PdistGradNet(p)
    output_ms = net(y_grad, x, y)
    return output_ms


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_pdist_grad_graph(dtype, eps):
    """
    Feature: test PdistGrad operation in result
    Description: test the Pdist p = 2.0
    Expectation: the output matches numpy
    """
    y_grad = Tensor(np.array([1., 1., 2.]).astype(dtype))
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]]).astype(dtype))
    y = Tensor(np.array([1.41421356, 2.82842712, 1.41421356]).astype(dtype))
    p = 2.0
    error = np.ones(shape=(3, 2)) * eps
    output_ms_graph = pdist_grad_graph(y_grad, x, y, p)
    out_pt = np.array([[-1.41421356, -1.41421356], [-0.70710678, -0.70710678], [2.12132034, 2.12132034]]).astype(dtype)
    diff_graph = np.abs(output_ms_graph.asnumpy() - out_pt)
    assert np.all(diff_graph < error)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_pdist_grad_pynative(dtype, eps):
    """
    Feature: test PdistGrad operation in result
    Description: test the Pdist p = 2.0
    Expectation: the output matches numpy
    """
    y_grad = Tensor(np.array([1., 1., 2.]).astype(dtype))
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]]).astype(dtype))
    y = Tensor(np.array([1.41421356, 2.82842712, 1.41421356]).astype(dtype))
    p = 2.0
    error = np.ones(shape=(3, 2)) * eps
    output_ms_pynative = pdist_grad_pynative(y_grad, x, y, p)
    out_pt = np.array([[-1.41421356, -1.41421356], [-0.70710678, -0.70710678], [2.12132034, 2.12132034]]).astype(dtype)
    diff_pynative = np.abs(output_ms_pynative.asnumpy() - out_pt)
    assert np.all(diff_pynative < error)
