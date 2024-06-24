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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.common.dtype as mstype
import mindspore.context as context
from mindspore import Tensor, ops
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


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('start_np, stop_np', [(5, 150), (-25, 147), (-25.3, -147)])
@pytest.mark.parametrize('num_np', [1, 12, 10, 20])
def test_lin_space(start_np, stop_np, num_np):
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    num = num_np
    result_ms = ops.linspace(start, stop, num).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('start_np, stop_np', [(5, 150), (-25, 147), (-25.3, -147)])
@pytest.mark.parametrize('num_np', [10, 20, 36])
def test_lin_space_net(start_np, stop_np, num_np):
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net = LinSpaceNet(num_np)
    result_ms = net(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lin_space_vmap_1d():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start_np = np.random.randn(5)
    stop_np = np.random.randn(5)
    num_np = 10

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net = LinSpaceNet(num_np)
    result_ms = ops.vmap(net, (0, 0))(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lin_space_vmap_2d():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start_np = np.random.randn(5, 4)
    stop_np = np.random.randn(4, 5)
    num_np = 10

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net = LinSpaceNet(num_np)
    result_ms = ops.vmap(ops.vmap(net, (0, 0)), (1, 0))(start, stop).asnumpy()

    start_np = np.moveaxis(start_np, 1, 0)
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lin_space_vmap_dynamic_shape():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start_np = np.random.randn(5)
    stop_np = np.random.randn(5)
    num_np = 10

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)

    dynamic_net = LinSpaceNet(num_np)
    place_holder = Tensor(shape=[None], dtype=mstype.float32)
    dynamic_net.set_inputs(place_holder, place_holder)

    result_ms = ops.vmap(dynamic_net, (0, 0))(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_lin_space_num():
    """
    Feature: ALL To ALL
    Description: test cases for LinSpace Net
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    np.random.seed(0)

    start_np = np.random.randn(5)
    stop_np = np.random.randn(5)
    num_np = 1

    start = Tensor(start_np, dtype=mstype.float32)
    stop = Tensor(stop_np, dtype=mstype.float32)
    net = LinSpaceNet(num_np)
    result_ms = ops.vmap(net, (0, 0))(start, stop).asnumpy()
    result_np = np.linspace(start_np, stop_np, num_np, axis=-1)
    assert np.allclose(result_ms, result_np)
