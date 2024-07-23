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
import numpy as np
import pytest

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from tests.mark_utils import arg_mark


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.expand_dims = P.ExpandDims()

    def construct(self, tensor):
        return self.expand_dims(tensor, -1)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64])
def test_sqeeze_net_ascend(data_type):
    """
    Feature: Test ExpandDims DynamicShape.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    x_np = np.random.randn(1, 16, 1, 1).astype(data_type)
    x = Tensor(x_np)

    net = Net()
    input_dyn = Tensor(shape=[1, None, 1, 1], dtype=x.dtype)
    net.set_inputs(input_dyn)

    expected = np.expand_dims(x_np, -1)

    context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="Ascend")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_sqeeze_net_cpu(data_type):
    """
    Feature: Test ExpandDims DynamicShape.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    x_np = np.random.randn(1, 16, 1, 1).astype(data_type)
    x = Tensor(x_np)

    net = Net()
    input_dyn = Tensor(shape=[1, None, 1, 1], dtype=x.dtype)
    net.set_inputs(input_dyn)

    expected = np.expand_dims(x_np, -1)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)


@arg_mark(plat_marks=['platform_gpu'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_sqeeze_net_gpu(data_type):
    """
    Feature: Test ExpandDims DynamicShape.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    x_np = np.random.randn(1, 16, 1, 1).astype(data_type)
    x = Tensor(x_np)

    net = Net()
    input_dyn = Tensor(shape=[1, None, 1, 1], dtype=x.dtype)
    net.set_inputs(input_dyn)

    expected = np.expand_dims(x_np, -1)

    context.set_context(mode=context.GRAPH_MODE, device_target="GPU")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="GPU")
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == expected)
