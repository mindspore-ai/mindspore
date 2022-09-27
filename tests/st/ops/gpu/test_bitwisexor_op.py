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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor
from mindspore.ops import operations as P


class NetBitwiseXor(nn.Cell):
    def __init__(self):
        super(NetBitwiseXor, self).__init__()
        self.bitwisexor = P.BitwiseXor()

    def construct(self, x1, x2):
        return self.bitwisexor(x1, x2)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwisexor_graph():
    """
    Description: What input in what scene
    Expectation:int32test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]).astype(np.int32))
        x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]).astype(np.int32))
        result_expect = np.bitwise_xor([0, 0, 1, -1, 1, 1, 1], [0, 1, 1, -1, -1, 2, 3])
        net = NetBitwiseXor()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwisexor_graph1():
    """
    Description: What input in what scene
    Expectation:int16test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, -1, 1, 1, 1]).astype(np.int16))
        x2 = Tensor(np.array([0, 1, 1, -1, -1, 2, 3]).astype(np.int16))
        result_expect = np.bitwise_xor([0, 0, 1, -1, 1, 1, 1], [0, 1, 1, -1, -1, 2, 3])
        net = NetBitwiseXor()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)


@pytest.mark.level1
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_bitwisexor_graph2():
    """
    Description: What input in what scene
    Expectation:uint16test
    """
    for mode in [context.PYNATIVE_MODE, context.GRAPH_MODE]:
        context.set_context(mode=mode, device_target="GPU")

        x1 = Tensor(np.array([0, 0, 1, 1, 1, 1, 1]).astype(np.uint16))
        x2 = Tensor(np.array([0, 1, 1, 1, 5, 2, 3]).astype(np.uint16))
        result_expect = np.bitwise_xor([0, 0, 1, 1, 1, 1, 1], [0, 1, 1, 1, 5, 2, 3])
        net = NetBitwiseXor()
        output = net(x1, x2)
        result = output.asnumpy()
        eps = np.array([1e-6 for i in range(7)])
        assert np.all(abs(result_expect - result) < eps)
