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
from mindspore.ops.operations import nn_ops as P


class PdistNet(nn.Cell):
    def __init__(self, p=2.0):
        super().__init__()
        self.pdist = P.Pdist(p=p)

    def construct(self, x):
        return self.pdist(x)


def pdist_graph(x, p):
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    net = PdistNet(p)
    output_ms = net(x)
    return output_ms


def pdist_pynative(x, p):
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    net = PdistNet(p)
    output_ms = net(x)
    return output_ms


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_pdist_graph(dtype, eps):
    """
    Feature: test Pdist operation in result
    Description: test the Pdist p = 2.0
    Expectation: the output matches numpy
    """
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]]).astype(dtype))
    error = np.ones(shape=(3,)) * eps
    p = 2.0
    output_ms_graph = pdist_graph(x, p)
    out_expect = np.array([1.41421356, 2.82842712, 1.41421356]).astype(dtype)
    diff_graph = np.abs(output_ms_graph.asnumpy() - out_expect)
    assert np.all(diff_graph < error)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype, eps', [(np.float32, 1.0e-4), (np.float64, 1.0e-5)])
def test_pdist_pynative(dtype, eps):
    """
    Feature: test Pdist operation in result
    Description: test the Pdist p = 2.0
    Expectation: the output matches numpy
    """
    x = Tensor(np.array([[1., 1.], [2., 2.], [3., 3.]]).astype(dtype))
    error = np.ones(shape=(3,)) * eps
    p = 2.0
    output_ms_pynative = pdist_pynative(x, p)
    out_expect = np.array([1.41421356, 2.82842712, 1.41421356]).astype(dtype)
    diff_pynative = np.abs(output_ms_pynative.asnumpy() - out_expect)
    assert np.all(diff_pynative < error)
