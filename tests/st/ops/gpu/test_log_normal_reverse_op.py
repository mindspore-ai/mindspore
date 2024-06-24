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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.nn as nn
from mindspore import Tensor
import mindspore.context as context
import mindspore.ops.operations.random_ops as P


class NetLogNormalReverse(nn.Cell):
    def __init__(self, input_mean_=1.0, input_std_=2.0):
        super(NetLogNormalReverse, self).__init__()
        self.log_normal_reverse = P.LogNormalReverse(input_mean_, input_std_)

    def construct(self, x):
        return self.log_normal_reverse(x)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_log_normal_reverse_float16_4d():
    """
    Feature: LogNormalReverse gpu TEST.
    Description: 4d - float16 test case for LogNormalReverse
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.random.randn(9, 8, 7, 6).astype(np.float16))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (9, 8, 7, 6)
    assert output.shape == expect

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.random.randn(9, 8, 7, 6).astype(np.float16))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (9, 8, 7, 6)
    assert output.shape == expect


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_log_normal_reverse_float32_3d():
    """
    Feature: LogNormalReverse gpu TEST.
    Description: 3d - float32 test case for LogNormalReverse
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.random.randn(10, 10, 10).astype(np.float32))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (10, 10, 10)
    assert output.shape == expect

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.random.randn(10, 10, 10).astype(np.float32))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (10, 10, 10)
    assert output.shape == expect


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_log_normal_reverse_float64_2d():
    """
    Feature: LogNormalReverse gpu TEST.
    Description: 2d - float64 test case for LogNormalReverse
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.random.randn(20, 20).astype(np.float64))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (20, 20)
    assert output.shape == expect

    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    x = Tensor(np.random.randn(20, 20).astype(np.float64))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    output = log_normal_reverse(x)
    expect = (20, 20)
    assert output.shape == expect


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_log_normal_reverse_invalid_input_type():
    """
    Feature: LogNormalReverse gpu TEST.
    Description: Test running the op with int8's in the init type in incorrect positions.
    Expectation: Expected to fail.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    x = Tensor(np.random.randn(3, 4, 2, 5).astype(np.int8))
    mean = 1.0
    std = 1.0
    log_normal_reverse = NetLogNormalReverse(mean, std)
    with pytest.raises(TypeError):
        log_normal_reverse(x)
