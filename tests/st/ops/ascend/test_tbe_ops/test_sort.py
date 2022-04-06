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
import mindspore
from mindspore import Tensor, ops, nn, context

np.random.seed(5)


class NetCPU(nn.Cell):
    def __init__(self):
        super(NetCPU, self).__init__()
        self.sort = ops.Sort(axis=1)

    def construct(self, x):
        y = self.sort(x)
        return y


class NetAscend(nn.Cell):
    def __init__(self):
        super(NetAscend, self).__init__()
        self.sort = ops.Sort(axis=1)

    def construct(self, x):
        y = self.sort(x)
        return y


def cpu(tensor, mode):
    context.set_context(mode=mode, device_target="CPU")
    net = NetCPU()
    y = net(tensor)
    return y


def ascend(tensor, mode):
    context.set_context(mode=mode, device_target="Ascend")
    net = NetAscend()
    y = net(tensor)
    return y


def test_sort_with_axis_graph_mode():
    """
    Feature: sort op support the axis value not -1 with graph mode.
    Description: sort op support the axis value not -1.
    Expectation: same as the calculation result on CPU.
    """
    tensor = Tensor(np.random.random([3, 7, 7, 2]), mindspore.float16)
    cpu_out = cpu(tensor, context.GRAPH_MODE)
    ascend_out = ascend(tensor, context.GRAPH_MODE)
    assert np.allclose(cpu_out[0].asnumpy(), ascend_out[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(cpu_out[1].asnumpy(), ascend_out[1].asnumpy(), 0, 0)


def test_sort_with_axis_pynative_mode():
    """
    Feature: sort op support the axis value not -1 with pynative mode.
    Description: sort op support the axis value not -1.
    Expectation: same as the calculation result on CPU.
    """
    tensor = Tensor(np.random.random([3, 7, 7, 2]), mindspore.float16)
    cpu_out = cpu(tensor, context.PYNATIVE_MODE)
    ascend_out = ascend(tensor, context.PYNATIVE_MODE)
    assert np.allclose(cpu_out[0].asnumpy(), ascend_out[0].asnumpy(), 0.00001, 0.00001)
    assert np.allclose(cpu_out[1].asnumpy(), ascend_out[1].asnumpy(), 0, 0)
