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
from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.ops.composite import GradOperation
from mindspore.ops.operations import _inner_ops as inner

class Grad(nn.Cell):
    def __init__(self, network):
        super(Grad, self).__init__()
        self.grad = GradOperation(get_all=True, sens_param=True)
        self.network = network

    def construct(self, input_x, dout):
        return self.grad(self.network)(input_x, dout)


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.HSigmoid = P.HSigmoid()

    def construct(self, x):
        return self.HSigmoid(x)


class DynamicNet(nn.Cell):
    def __init__(self):
        super(DynamicNet, self).__init__()
        self.HSigmoid = P.HSigmoid()
        self.d = inner.GpuConvertToDynamicShape()

    def construct(self, x):
        x = self.d(x)
        return self.HSigmoid(x)


def generate_testcases(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([-1, -2, 0, 4, 5]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.array([0.33333334, 0.16666667, 0.5, 1, 1]).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)

    sens = np.array([-1.45, 0.63, 0.34, 6.43, 34.6]).astype(nptype)
    backward_net = Grad(Net())
    output = backward_net(Tensor(x), Tensor(sens))
    expect = np.array([-0.2416667, 0.1049999, 5.66666685e-02, 0, 0]).astype(nptype)
    np.testing.assert_almost_equal(output[0].asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([-1, -2, 0, 4, 5]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.array([0.33333334, 0.16666667, 0.5, 1, 1]).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)

    sens = np.array([-1.45, 0.63, 0.34, 6.43, 34.6]).astype(nptype)
    backward_net = Grad(Net())
    output = backward_net(Tensor(x), Tensor(sens))
    expect = np.array([-0.2416667, 0.1049999, 5.66666685e-02, 0, 0]).astype(nptype)
    np.testing.assert_almost_equal(output[0].asnumpy(), expect)


def generate_dynamic_testcase(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([-1, -2, 0, 2, 1]).astype(nptype)
    net = DynamicNet()
    output = net(Tensor(x))
    expect = np.array([0.33333334, 0.16666667, 0.5, 0.8333333, 0.6666667]).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hsigmoid_dynamic_float32():
    generate_dynamic_testcase(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hsigmoid_float32():
    generate_testcases(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_hsigmoid_float16():
    generate_testcases(np.float16)
