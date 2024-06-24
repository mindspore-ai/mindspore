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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


class NetBitwiseOr(nn.Cell):
    def __init__(self):
        super(NetBitwiseOr, self).__init__()
        self.bitwiseor = P.BitwiseOr()

    def construct(self, x1, x2):
        return self.bitwiseor(x1, x2)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bitwiseor_graph():
    """
    Description: What input in what scene
    Expectation:uint16test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]).astype(np.int32))
        x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]).astype(np.int32))
        result_expect = np.bitwise_or([0, 0, 1, -1, 1, 1, 1], [0, 1, 1, -1, -1, 2, 3])
        net = NetBitwiseOr()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bitwiseor_graph2():
    """
    Description: What input in what scene
    Expectation:uint16test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]).astype(np.int16))
        x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]).astype(np.int16))
        result_expect = np.bitwise_or([0, 0, 1, -1, 1, 1, 1], [0, 1, 1, -1, -1, 2, 3])
        net = NetBitwiseOr()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_bitwiseor_graph3():
    """
    Description: What input in what scene
    Expectation:uint16test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, 1, 1, 1, 1]).astype(np.uint16))
        x2 = Tensor(np.array([0, 1, 1, 1, 5, 2, 3]).astype(np.uint16))
        result_expect = np.bitwise_or([0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 5, 2, 3])
        net = NetBitwiseOr()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)
