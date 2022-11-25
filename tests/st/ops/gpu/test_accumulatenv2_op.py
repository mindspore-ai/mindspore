# Copyright 2019-2021 Huawei Technologies Co., Ltd
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
import mindspore
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common.api import ms_function
from mindspore.ops import operations as P


class NetAccumulateNV2(nn.Cell):
    def __init__(self):
        super(NetAccumulateNV2, self).__init__()
        self.accumulatenv2 = P.AccumulateNV2()

    @ms_function
    def construct(self, *x):
        return self.accumulatenv2(x)


def pynative_net():
    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    net = NetAccumulateNV2()
    x = Tensor(np.array([1, 1, 1, 1]), mindspore.float32)
    y = Tensor(np.array([3, 3, 3, 3]), mindspore.float32)
    output = net(x, y, x, y, x, y, y, y, y)
    expect_result = [21, 21, 21, 21]
    assert (output.asnumpy() == expect_result).all()


def graph_net():
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    net = NetAccumulateNV2()
    m = Tensor(np.array([2, 2, 2]), mindspore.float64)
    n = Tensor(np.array([3, 3, 3]), mindspore.float64)
    output = net(m, n, m, n)
    expect_result = [10, 10, 10]
    assert (output.asnumpy() == expect_result).all()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_accumulate_n_v2_pynative_float32():
    """
    Feature: ALL To ALL
    Description: test cases for AccumulateNV2
    Expectation: the result match to tensorflow
    """
    pynative_net()


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_accumulate_n_v2_graph_float64():
    """
    Feature: ALL To ALL
    Description: test cases for AccumulateNV2
    Expectation: the result match to tensorflow
    """
    graph_net()
