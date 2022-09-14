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

import mindspore.context as context
import mindspore.nn as nn
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class SubNet(nn.Cell):
    def __init__(self):
        super(SubNet, self).__init__()
        self.sub = P.Sub()

    def construct(self, x, y):
        return self.sub(x, y)


class DivNet(nn.Cell):
    def __init__(self):
        super(DivNet, self).__init__()
        self.div = P.Div()

    def construct(self, x, y):
        return self.div(x, y)


class FloorDivNet(nn.Cell):
    def __init__(self):
        super(FloorDivNet, self).__init__()
        self.floor_div = P.FloorDiv()

    def construct(self, x, y):
        return self.floor_div(x, y)


class ModNet(nn.Cell):
    def __init__(self):
        super(ModNet, self).__init__()
        self.mod = P.Mod()

    def construct(self, x, y):
        return self.mod(x, y)


class FloorModNet(nn.Cell):
    def __init__(self):
        super(FloorModNet, self).__init__()
        self.floor_mod = P.FloorMod()

    def construct(self, x, y):
        return self.floor_mod(x, y)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_sub():
    """
    Feature: ALL To ALL
    Description: test cases for Sub.
    Expectation: the result match to numpy
    """
    x = np.random.rand(2, 3, 4, 4).astype(np.float32)
    y = np.random.rand(4, 1).astype(np.float32)
    net = SubNet()
    output = net(Tensor(x), Tensor(y, mindspore.float32))
    expect_output = x - y
    assert np.all(output.asnumpy() == expect_output)

    # float64
    x = np.random.rand(2, 3, 4, 4).astype(np.float64)
    y = np.random.rand(4, 1).astype(np.float64)
    net = SubNet()
    output = net(Tensor(x), Tensor(y, mindspore.float64))
    expect_output = x - y
    assert np.all(output.asnumpy() == expect_output)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_div():
    """
    Feature: ALL To ALL
    Description: test cases for Div.
    Expectation: the result match to numpy
    """
    prop = 1 if np.random.random() < 0.5 else -1
    x0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x1_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y1_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float32) * prop
    x2_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.float16) * prop
    y2_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float16) * prop
    x3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    y3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    x4_np = np.array(768).astype(np.float32) * prop
    y4_np = np.array(3072.5).astype(np.float32) * prop
    x5_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int32) * prop
    y5_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    x6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    y6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x7_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int64) * prop
    y7_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int64) * prop
    x8_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float64) * prop
    y8_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float64) * prop

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)
    x5 = Tensor(x5_np)
    y5 = Tensor(y5_np)
    x6 = Tensor(x6_np)
    y6 = Tensor(y6_np)
    x7 = Tensor(x7_np)
    y7 = Tensor(y7_np)
    x8 = Tensor(x8_np)
    y8 = Tensor(y8_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    div = DivNet()
    output0 = div(x0, y0)
    expect0 = np.divide(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = div(x1, y1)
    expect1 = np.divide(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = div(x2, y2)
    expect2 = np.divide(x2_np, y2_np).astype(np.float16)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    output3 = div(x3, y3)
    expect3 = np.divide(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = div(x4, y4)
    expect4 = np.divide(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    output5 = div(x5, y5)
    expect5 = x5_np // y5_np
    assert np.all(output5.asnumpy() == expect5)

    output6 = div(x6, y6)
    expect6 = np.divide(x6_np, y6_np)
    diff6 = output6.asnumpy() - expect6
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output6.shape == expect6.shape

    output7 = div(x7, y7)
    expect7 = np.divide(x7_np, y7_np).astype(np.int64)
    assert np.all(output7.asnumpy() == expect7)
    assert output7.shape == expect7.shape

    output8 = div(x8, y8)
    expect8 = np.divide(x8_np, y8_np)
    diff8 = output8.asnumpy() - expect8
    error8 = np.ones(shape=expect8.shape) * 1.0e-7
    assert np.all(diff8 < error8)
    assert output8.shape == expect8.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor_div():
    """
    Feature: ALL To ALL
    Description: test cases for FloorDiv.
    Expectation: the result match to numpy
    """
    prop = 1 if np.random.random() < 0.5 else -1
    x0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y0_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float32) * prop
    x1_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.float16) * prop
    y1_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float16) * prop
    x2_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int32) * prop
    y2_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    x3_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    y3_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x4_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int64) * prop
    y4_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int64) * prop
    x5_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float64) * prop
    y5_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float64) * prop

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)
    x5 = Tensor(x5_np)
    y5 = Tensor(y5_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    floor_div = FloorDivNet()
    output0 = floor_div(x0, y0)
    expect0 = np.floor_divide(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = floor_div(x1, y1)
    expect1 = np.floor_divide(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = floor_div(x2, y2)
    expect2 = np.floor_divide(x2_np, y2_np).astype(np.float16)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    output3 = floor_div(x3, y3)
    expect3 = np.floor_divide(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = floor_div(x4, y4)
    expect4 = np.floor_divide(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    output5 = floor_div(x5, y5)
    expect5 = np.floor_divide(x5_np, y5_np)
    diff5 = output5.asnumpy() - expect5
    error5 = np.ones(shape=expect5.shape) * 1.0e-7
    assert np.all(diff5 < error5)
    assert output5.shape == expect5.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor_div_cpu_dynamic_shape():
    """
    Feature: test FloorDiv op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = FloorDivNet()
    x_dyn = Tensor(shape=[None, 32], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[16, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn, y_dyn)
    x = np.random.randn(16, 32)
    y = np.random.randn(16, 32)
    output = net(Tensor(x, mindspore.float32), Tensor(y, mindspore.float32))
    expect_shape = (16, 32)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mod():
    """
    Feature: ALL To ALL
    Description: test cases for Mod.
    Expectation: the result match to numpy
    """
    prop = 1 if np.random.random() < 0.5 else -1
    x0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x1_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y1_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float32) * prop
    x2_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.float16) * prop
    y2_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float16) * prop
    x3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    y3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    x4_np = np.array(768).astype(np.float32) * prop
    y4_np = np.array(3072.5).astype(np.float32) * prop
    x5_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int32) * prop
    y5_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    x6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    y6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x7_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int64) * prop
    y7_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int64) * prop

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)
    x5 = Tensor(x5_np)
    y5 = Tensor(y5_np)
    x6 = Tensor(x6_np)
    y6 = Tensor(y6_np)
    x7 = Tensor(x7_np)
    y7 = Tensor(y7_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    mod = ModNet()
    output0 = mod(x0, y0)
    expect0 = np.mod(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = mod(x1, y1)
    expect1 = np.mod(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = mod(x2, y2)
    expect2 = np.mod(x2_np, y2_np).astype(np.float16)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    output3 = mod(x3, y3)
    expect3 = np.mod(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = mod(x4, y4)
    expect4 = np.mod(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    output5 = mod(x5, y5)
    expect5 = np.mod(x5_np, y5_np)
    assert np.all(output5.asnumpy() == expect5)
    assert output5.shape == expect5.shape

    output6 = mod(x6, y6)
    expect6 = np.mod(x6_np, y6_np)
    diff6 = output6.asnumpy() - expect6
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output6.shape == expect6.shape

    output7 = mod(x7, y7)
    expect7 = np.mod(x7_np, y7_np).astype(np.int64)
    assert np.all(output7.asnumpy() == expect7)
    assert output6.shape == expect6.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_mod_cpu_dynamic_shape():
    """
    Feature: test Mod op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = ModNet()
    x_dyn = Tensor(shape=[None, 32], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[16, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn, y_dyn)
    x = np.random.randn(16, 32)
    y = np.random.randn(16, 32)
    output = net(Tensor(x, mindspore.float32), Tensor(y, mindspore.float32))
    expect_shape = (16, 32)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor_mod():
    """
    Feature: ALL To ALL
    Description: test cases for FloorMod.
    Expectation: the result match to numpy
    """
    prop = 1 if np.random.random() < 0.5 else -1
    x0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y0_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x1_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    y1_np = np.random.randint(1, 100, (2, 1, 4, 4)).astype(np.float32) * prop
    x2_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.float16) * prop
    y2_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float16) * prop
    x3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    y3_np = np.random.randint(1, 100, 1).astype(np.float32) * prop
    x4_np = np.array(768).astype(np.float32) * prop
    y4_np = np.array(3072.5).astype(np.float32) * prop
    x5_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int32) * prop
    y5_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    x6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int32) * prop
    y6_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.float32) * prop
    x7_np = np.random.randint(1, 100, (2, 1, 1, 4)).astype(np.int64) * prop
    y7_np = np.random.randint(1, 100, (2, 3, 4, 4)).astype(np.int64) * prop

    x0 = Tensor(x0_np)
    y0 = Tensor(y0_np)
    x1 = Tensor(x1_np)
    y1 = Tensor(y1_np)
    x2 = Tensor(x2_np)
    y2 = Tensor(y2_np)
    x3 = Tensor(x3_np)
    y3 = Tensor(y3_np)
    x4 = Tensor(x4_np)
    y4 = Tensor(y4_np)
    x5 = Tensor(x5_np)
    y5 = Tensor(y5_np)
    x6 = Tensor(x6_np)
    y6 = Tensor(y6_np)
    x7 = Tensor(x7_np)
    y7 = Tensor(y7_np)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    floor_mod = FloorModNet()
    output0 = floor_mod(x0, y0)
    expect0 = np.mod(x0_np, y0_np)
    diff0 = output0.asnumpy() - expect0
    error0 = np.ones(shape=expect0.shape) * 1.0e-5
    assert np.all(diff0 < error0)
    assert output0.shape == expect0.shape

    output1 = floor_mod(x1, y1)
    expect1 = np.mod(x1_np, y1_np)
    diff1 = output1.asnumpy() - expect1
    error1 = np.ones(shape=expect1.shape) * 1.0e-5
    assert np.all(diff1 < error1)
    assert output1.shape == expect1.shape

    output2 = floor_mod(x2, y2)
    expect2 = np.mod(x2_np, y2_np).astype(np.float16)
    diff2 = output2.asnumpy() - expect2
    error2 = np.ones(shape=expect2.shape) * 1.0e-5
    assert np.all(diff2 < error2)
    assert output2.shape == expect2.shape

    output3 = floor_mod(x3, y3)
    expect3 = np.mod(x3_np, y3_np)
    diff3 = output3.asnumpy() - expect3
    error3 = np.ones(shape=expect3.shape) * 1.0e-5
    assert np.all(diff3 < error3)
    assert output3.shape == expect3.shape

    output4 = floor_mod(x4, y4)
    expect4 = np.mod(x4_np, y4_np)
    diff4 = output4.asnumpy() - expect4
    error4 = np.ones(shape=expect4.shape) * 1.0e-5
    assert np.all(diff4 < error4)
    assert output4.shape == expect4.shape

    output5 = floor_mod(x5, y5)
    expect5 = np.mod(x5_np, y5_np)
    assert np.all(output5.asnumpy() == expect5)
    assert output5.shape == expect5.shape

    output6 = floor_mod(x6, y6)
    expect6 = np.mod(x6_np, y6_np)
    diff6 = output6.asnumpy() - expect6
    error6 = np.ones(shape=expect6.shape) * 1.0e-5
    assert np.all(diff6 < error6)
    assert output6.shape == expect6.shape

    output7 = floor_mod(x7, y7)
    expect7 = np.mod(x7_np, y7_np).astype(np.int64)
    assert np.all(output7.asnumpy() == expect7)
    assert output6.shape == expect6.shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_floor_mod_cpu_dynamic_shape():
    """
    Feature: test FloorMod op in cpu.
    Description: test the ops in dynamic shape.
    Expectation: expect correct shape result.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = FloorModNet()
    x_dyn = Tensor(shape=[None, 32], dtype=mindspore.float32)
    y_dyn = Tensor(shape=[16, None], dtype=mindspore.float32)
    net.set_inputs(x_dyn, y_dyn)
    x = np.random.randn(16, 32)
    y = np.random.randn(16, 32)
    output = net(Tensor(x, mindspore.float32), Tensor(y, mindspore.float32))
    expect_shape = (16, 32)
    assert output.asnumpy().shape == expect_shape


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_dynamic_sub(dtype):
    """
    Feature: ALL To ALL
    Description: test cases for Sub dynamic shape.
    Expectation: the result match to numpy
    """
    context.set_context(mode=context.GRAPH_MODE)
    x = np.random.rand(2, 3, 4, 4).astype(dtype)
    y = np.random.rand(4, 1).astype(dtype)
    benchmark_output = x - y
    loss = 1e-5
    sub_net = SubNet()
    real_x = Tensor(x)
    real_y = Tensor(y)
    dy_x_shape = [None for _ in x.shape]
    dy_y_shape = [None for _ in y.shape]
    input_dyn_x = Tensor(shape=dy_x_shape, dtype=real_x.dtype)
    input_dyn_y = Tensor(shape=dy_y_shape, dtype=real_y.dtype)
    sub_net.set_inputs(input_dyn_x, input_dyn_y)
    ms_result = sub_net(real_x, real_y)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
    context.set_context(mode=context.PYNATIVE_MODE)
    ms_result = sub_net(real_x, real_y)
    np.testing.assert_allclose(benchmark_output, ms_result.asnumpy(), rtol=loss, atol=loss)
