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

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor, ops
from mindspore.ops import operations as P


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.expand_dims = P.ExpandDims()

    def construct(self, tensor):
        return self.expand_dims(tensor, -1)


class NetConstant(nn.Cell):
    def __init__(self, data):
        super(NetConstant, self).__init__()
        self.expand_dims = P.ExpandDims()
        self.tensor = Tensor(data)

    def construct(self):
        return self.expand_dims(self.tensor, -1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_net(data_type):
    """
    Feature: Test ExpandDims CPU.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.random.randn(1, 16, 1, 1).astype(data_type)
    net = Net()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_net_constant():
    """
    Feature: Test ExpandDims CPU.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)
    net = NetConstant(x)
    output = net()
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_func():
    """
    Feature: Test ExpandDims CPU.
    Description: Test functional api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)
    output = ops.expand_dims(Tensor(x), -1)
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_tensor():
    """
    Feature: Test ExpandDims CPU.
    Description: Test Tensor api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)
    output = Tensor(x).expand_dims(-1)
    assert np.all(output.asnumpy() == np.expand_dims(x, -1))
