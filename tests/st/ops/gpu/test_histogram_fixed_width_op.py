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

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations.math_ops import HistogramFixedWidth


class NetHistogramFixedWidth(nn.Cell):
    def __init__(self, nbins):
        super().__init__()
        self.histogramfixedwidth = HistogramFixedWidth(nbins=nbins)

    def construct(self, x, range_):
        return self.histogramfixedwidth(x, range_)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_histogramfixedwidth_1d():
    """
    Feature: HistogramFixedWidth gpu TEST.
    Description: 1d test case for HistogramFixedWidth
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")

    x_np = np.arange(0, 10).astype(np.int32)
    range_np = np.array([0, 10]).astype(np.int32)
    nbins = 5
    net = NetHistogramFixedWidth(nbins)
    x_ms = Tensor(x_np)
    range_ms = Tensor(range_np)
    output_ms = net(x_ms, range_ms)
    except_output_np = np.array([2, 2, 2, 2, 2]).astype(np.int32)
    assert np.allclose(except_output_np, output_ms.asnumpy())


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_histogramfixedwidth_2d():
    """
    Feature: HistogramFixedWidth gpu TEST.
    Description: 2d test case for HistogramFixedWidth
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x_np = np.array([np.arange(-5, -1), np.arange(4, 8)]).astype(np.float16)
    range_np = np.array([0, 10]).astype(np.float16)
    except_output_np = np.array([4, 0, 2, 2, 0]).astype(np.int32)

    nbins = 5
    x_ms = Tensor(x_np)
    range_ms = Tensor(range_np)
    net = NetHistogramFixedWidth(nbins)
    output_ms = net(x_ms, range_ms)
    assert np.allclose(except_output_np, output_ms.asnumpy())
