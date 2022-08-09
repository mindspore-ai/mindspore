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
from mindspore import Tensor
from mindspore.nn import Cell
import mindspore.ops as P
from mindspore.ops.functional import vmap


class SqueezeNet(Cell):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.squeeze = P.Squeeze()

    def construct(self, x):
        return self.squeeze(x)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type",
                         [np.bool, np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64,
                          np.uint64, np.float16, np.float32, np.float64, np.complex64, np.complex128])
def test_squeeze(data_type):
    """
    Feature: Test Squeeze CPU.
    Description: The input data type contains common valid types including bool
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(data_type)
    net = SqueezeNet()
    output = net(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_func():
    """
    Feature: Test Squeeze CPU.
    Description: Test functional api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)

    output = P.squeeze(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())

    output = P.squeeze(Tensor(x), 0)
    assert np.all(output.asnumpy() == x.squeeze(0))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_tensor():
    """
    Feature: Test Squeeze CPU.
    Description: Test Tensor api.
    Expectation: match to np benchmark.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 1).astype(np.int32)

    output = Tensor(x).squeeze()
    assert np.all(output.asnumpy() == x.squeeze())

    output = Tensor(x).squeeze(0)
    assert np.all(output.asnumpy() == x.squeeze(0))


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
def test_squeeze_vmap():
    """
    Feature: Test Squeeze CPU vmap.
    Description: test vmap for squeeze.
    Expectation: match to np benchmark.
    """

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    np.random.seed(0)
    x = np.random.randn(1, 16, 1, 16)
    net = SqueezeNet()
    output = vmap(net, in_axes=-1, out_axes=-1)(Tensor(x))
    assert np.all(output.asnumpy() == x.squeeze())
