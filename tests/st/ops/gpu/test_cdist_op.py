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
from mindspore.ops.operations import math_ops as P


class CdistNet(nn.Cell):
    def __init__(self, p=2.0):
        super(CdistNet, self).__init__()
        self.cdist = P.Cdist(p=p)

    def construct(self, x1, x2):
        return self.cdist(x1, x2)


def cdist_graph(x1, x2, p):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    print("cdist_graph")
    net = CdistNet(p)
    output_ms = net(x1, x2)
    return output_ms


def cdist_pynative(x1, x2, p):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = CdistNet(p)
    output_ms = net(x1, x2)
    return output_ms


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_cdist_p_graph(dtype, eps):
    """
    Feature: Cdist gpu kernel
    Description: test the Cdist p = 2.0.
    Expectation: the output matches numpy
    """
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    p = 2.0
    output = cdist_graph(x1, x2, p=p)
    expect = np.array(
        [[[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(dtype)
    assert ((output.asnumpy() - expect) < eps).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_cdist_p_pynative(dtype, eps):
    """
    Feature: Cdist gpu kernel
    Description: test the Cdist p = 2.0.
    Expectation: the output matches numpy
    """
    x1 = Tensor(np.array([[[1.0, 1.0], [2.0, 2.0]]]).astype(dtype))
    x2 = Tensor(np.array([[[3.0, 3.0], [3.0, 3.0]]]).astype(dtype))
    p = 2.0
    output = cdist_pynative(x1, x2, p=p)
    expect = np.array(
        [[[2.828427, 2.828427], [1.4142135, 1.4142135]]]).astype(dtype)
    assert ((output.asnumpy() - expect) < eps).all()
