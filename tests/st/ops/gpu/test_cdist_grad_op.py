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
from mindspore.ops.operations import _grad_ops as P


class CdistGradNet(nn.Cell):
    def __init__(self, p=2.0):
        super(CdistGradNet, self).__init__()
        self.cdistgrad = P.CdistGrad(p=p)

    def construct(self, grad, x1, x2, y):
        return self.cdistgrad(grad, x1, x2, y)


def cdist_grad_graph(grad, x1, x2, y, p):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = CdistGradNet(p)
    output_ms = net(grad, x1, x2, y)
    return output_ms


def cdist_grad_pynative(grad, x1, x2, y, p):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = CdistGradNet(p)
    output_ms = net(grad, x1, x2, y)
    return output_ms


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_cdist_grad_p_graph(dtype, eps):
    """
    Feature: Cdistgrad gpu kernel
    Description: test the Cdist p = 3.0.
    Expectation: the output matches numpy
    """

    grad = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    dist = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    output = cdist_grad_graph(grad, x1, x2, dist, p=3.0)
    expect = np.array(
        [[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]]).astype(dtype)
    assert ((output.asnumpy() - expect) < eps).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_cdist_grad_p_pynative(dtype, eps):
    """
    Feature: Cdistgrad gpu kernel
    Description: test the Cdist p = 3.0.
    Expectation: the output matches numpy
    """

    grad = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    dist = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    output = cdist_grad_pynative(grad, x1, x2, dist, p=3.0)
    expect = np.array(
        [[[-0.8888889, -0.8888889], [-0.44444445, -0.44444445]]]).astype(dtype)
    assert ((output.asnumpy() - expect) < eps).all()
