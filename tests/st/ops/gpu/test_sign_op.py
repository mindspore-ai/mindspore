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
from mindspore import Tensor, ops


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.sign = ops.Sign()

    def construct(self, x):
        return self.sign(x)


def generate_testcases(nptype):
    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    x = np.array([2.0, 0.0, -1.0]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.array([1.0, 0.0, -1.0]).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    x = np.array([2.0, 0.0, -1.0]).astype(nptype)
    net = Net()
    output = net(Tensor(x))
    expect = np.array([1.0, 0.0, -1.0]).astype(nptype)
    np.testing.assert_almost_equal(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sign_int32():
    generate_testcases(np.int32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sign_float32():
    generate_testcases(np.float32)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sign_float16():
    generate_testcases(np.float16)
