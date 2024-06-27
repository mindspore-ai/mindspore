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
from tests.mark_utils import arg_mark
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P


class LessNet(nn.Cell):
    def __init__(self):
        super(LessNet, self).__init__()
        self.ops = P.Less()

    def construct(self, x, y):
        return self.ops(x, y)


class GreaterNet(nn.Cell):
    def __init__(self):
        super(GreaterNet, self).__init__()
        self.ops = P.Greater()

    def construct(self, x, y):
        return self.ops(x, y)


class LessEqualNet(nn.Cell):
    def __init__(self):
        super(LessEqualNet, self).__init__()
        self.ops = P.LessEqual()

    def construct(self, x, y):
        return self.ops(x, y)


class GreaterEqualNet(nn.Cell):
    def __init__(self):
        super(GreaterEqualNet, self).__init__()
        self.ops = P.GreaterEqual()

    def construct(self, x, y):
        return self.ops(x, y)


def gen_data():
    # Generate data which contains broadcast scene and two inputs are expr.
    np.random.seed(0)
    x0_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float32)
    y0_np = np.random.randint(1, 5, (2, 1, 4, 4)).astype(np.float32)
    x1_np = np.random.randint(1, 5, (2, 1, 1, 4)).astype(np.float16)
    y1_np = np.random.randint(1, 5, (2, 3, 4, 4)).astype(np.float16)
    x2_np = np.random.randint(1, 5, 1).astype(np.int32)
    y2_np = np.random.randint(1, 5, 1).astype(np.int32)
    x3_np = np.array([768]).astype(np.float32)
    y3_np = np.array([3072.5]).astype(np.float32)

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    return x0, y0, x1, y1, x2, y2, x3, y3


def get_less_net_output(x0, y0, x1, y1, x2, y2, x3, y3, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net_less = LessNet()
    less_output_0 = net_less(x0, y0).asnumpy()
    less_output_1 = net_less(x1, y1).asnumpy()
    less_output_2 = net_less(x2, y2).asnumpy()
    less_output_3 = net_less(x3, y3).asnumpy()
    return less_output_0, less_output_1, less_output_2, less_output_3


def get_greater_net_output(x0, y0, x1, y1, x2, y2, x3, y3, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net_greater = GreaterNet()
    greater_output_0 = net_greater(x0, y0).asnumpy()
    greater_output_1 = net_greater(x1, y1).asnumpy()
    greater_output_2 = net_greater(x2, y2).asnumpy()
    greater_output_3 = net_greater(x3, y3).asnumpy()
    return greater_output_0, greater_output_1, greater_output_2, greater_output_3


def get_less_equal_net_output(x0, y0, x1, y1, x2, y2, x3, y3, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net_less_equal = LessEqualNet()
    less_equal_output_0 = net_less_equal(x0, y0).asnumpy()
    less_equal_output_1 = net_less_equal(x1, y1).asnumpy()
    less_equal_output_2 = net_less_equal(x2, y2).asnumpy()
    less_equal_output_3 = net_less_equal(x3, y3).asnumpy()
    return less_equal_output_0, less_equal_output_1, less_equal_output_2, less_equal_output_3


def get_greater_equal_net_output(x0, y0, x1, y1, x2, y2, x3, y3, enable_graph_kernel=False):
    context.set_context(enable_graph_kernel=enable_graph_kernel)
    net_greater_equal = GreaterEqualNet()
    greter_equal_output_0 = net_greater_equal(x0, y0).asnumpy()
    greter_equal_output_1 = net_greater_equal(x1, y1).asnumpy()
    greter_equal_output_2 = net_greater_equal(x2, y2).asnumpy()
    greter_equal_output_3 = net_greater_equal(x3, y3).asnumpy()
    return greter_equal_output_0, greter_equal_output_1, greter_equal_output_2, greter_equal_output_3


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_net():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0, y0, x1, y1, x2, y2, x3, y3 = gen_data()
    out_gk_on_0, out_gk_on_1, out_gk_on_2, out_gk_on_3 = get_less_net_output(x0, y0, x1, y1, x2, y2, x3, y3, True)
    out_gk_off_0, out_gk_off_1, out_gk_off_2, out_gk_off_3 = get_less_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, False)

    assert np.all(out_gk_on_0 == out_gk_off_0)
    assert out_gk_on_0.shape == out_gk_off_0.shape
    assert np.all(out_gk_on_1 == out_gk_off_1)
    assert out_gk_on_1.shape == out_gk_off_1.shape
    assert np.all(out_gk_on_2 == out_gk_off_2)
    assert out_gk_on_2.shape == out_gk_off_2.shape
    assert np.all(out_gk_on_3 == out_gk_off_3)
    assert out_gk_on_3.shape == out_gk_off_3.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_greater_net():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0, y0, x1, y1, x2, y2, x3, y3 = gen_data()
    out_gk_on_0, out_gk_on_1, out_gk_on_2, out_gk_on_3 = get_greater_net_output(x0, y0, x1, y1, x2, y2, x3, y3, True)
    out_gk_off_0, out_gk_off_1, out_gk_off_2, out_gk_off_3 = get_greater_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, False)

    assert np.all(out_gk_on_0 == out_gk_off_0)
    assert out_gk_on_0.shape == out_gk_off_0.shape
    assert np.all(out_gk_on_1 == out_gk_off_1)
    assert out_gk_on_1.shape == out_gk_off_1.shape
    assert np.all(out_gk_on_2 == out_gk_off_2)
    assert out_gk_on_2.shape == out_gk_off_2.shape
    assert np.all(out_gk_on_3 == out_gk_off_3)
    assert out_gk_on_3.shape == out_gk_off_3.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_less_equal_net():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0, y0, x1, y1, x2, y2, x3, y3 = gen_data()
    out_gk_on_0, out_gk_on_1, out_gk_on_2, out_gk_on_3 = get_less_equal_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, True)
    out_gk_off_0, out_gk_off_1, out_gk_off_2, out_gk_off_3 = get_less_equal_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, False)

    assert np.all(out_gk_on_0 == out_gk_off_0)
    assert out_gk_on_0.shape == out_gk_off_0.shape
    assert np.all(out_gk_on_1 == out_gk_off_1)
    assert out_gk_on_1.shape == out_gk_off_1.shape
    assert np.all(out_gk_on_2 == out_gk_off_2)
    assert out_gk_on_2.shape == out_gk_off_2.shape
    assert np.all(out_gk_on_3 == out_gk_off_3)
    assert out_gk_on_3.shape == out_gk_off_3.shape


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_greater_equal_net():
    """
    Feature: todo
    Description: todo
    Expectation: todo
    """
    context.set_context(mode=context.GRAPH_MODE)
    x0, y0, x1, y1, x2, y2, x3, y3 = gen_data()
    out_gk_on_0, out_gk_on_1, out_gk_on_2, out_gk_on_3 = get_greater_equal_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, True)
    out_gk_off_0, out_gk_off_1, out_gk_off_2, out_gk_off_3 = get_greater_equal_net_output(
        x0, y0, x1, y1, x2, y2, x3, y3, False)

    assert np.all(out_gk_on_0 == out_gk_off_0)
    assert out_gk_on_0.shape == out_gk_off_0.shape
    assert np.all(out_gk_on_1 == out_gk_off_1)
    assert out_gk_on_1.shape == out_gk_off_1.shape
    assert np.all(out_gk_on_2 == out_gk_off_2)
    assert out_gk_on_2.shape == out_gk_off_2.shape
    assert np.all(out_gk_on_3 == out_gk_off_3)
    assert out_gk_on_3.shape == out_gk_off_3.shape
