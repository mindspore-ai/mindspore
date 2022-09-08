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

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore.common.tensor import Tensor
from mindspore.nn import Cell
from mindspore.ops import operations as P


class LinSpaceNet(Cell):
    def __init__(self, num):
        super(LinSpaceNet, self).__init__()
        self.ls_op = P.LinSpace()
        self.num = num

    def construct(self, start, stop):
        output = self.ls_op(start, stop, self.num)
        return output


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_space_1():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    start_np = 5
    stop_np = 150
    num_np = 12
    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    num = num_np
    ls_op = P.LinSpace()
    result_ms = ls_op(start, stop, num).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_shape_2():
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    start_np = -25
    stop_np = 147
    num_np = 10
    start = Tensor(start_np, dtype=mstype.float64)
    stop = Tensor(stop_np, dtype=mstype.float64)
    num = num_np
    ls_op = P.LinSpace()
    result_ms = ls_op(start, stop, num).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_shape_3():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    start_np = 25
    stop_np = -147
    num_np = 20
    start = Tensor(start_np, dtype=mstype.float64)
    stop = Tensor(stop_np, dtype=mstype.float64)
    net = LinSpaceNet(num_np)
    result_ms = net(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_shape_4():
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    start_np = -25.3
    stop_np = -147
    num_np = 36
    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net = LinSpaceNet(num_np)
    result_ms = net(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_space_dynamic_shape_1():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    start_np = -25.3
    stop_np = -147
    num_np = 36
    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)

    place_holder = Tensor(shape=[None], dtype=mstype.float32)
    dynamic_net = LinSpaceNet(num_np)
    dynamic_net.set_inputs(place_holder, place_holder)
    result_ms = dynamic_net(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_lin_space_dynamic_shape_2():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target='GPU')
    start_np = -25.3
    stop_np = -147
    num_np = 36
    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)

    place_holder = Tensor(shape=[None], dtype=mstype.float32)
    dynamic_net = LinSpaceNet(num_np)
    dynamic_net.set_inputs(place_holder, place_holder)
    result_ms = dynamic_net(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np)
    assert np.allclose(result_ms, result_np)
