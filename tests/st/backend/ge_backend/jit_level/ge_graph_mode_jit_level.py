# Copyright 2022-2023 Huawei Technologies Co., Ltd
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

import pytest  # pylint: disable=unused-import
import numpy as np

from mindspore import nn
from mindspore import ops
from mindspore import context, Tensor
from mindspore.common import JitConfig


class NetInner(nn.Cell):
    def __init__(self):
        super(NetInner, self).__init__()
        self.addn = ops.AddN()

    def construct(self, x, y):
        output = self.addn((x, y))
        return output


class NetOuter(nn.Cell):
    def __init__(self):
        super(NetOuter, self).__init__()
        self.addn = ops.AddN()
        self.inner_o3 = NetInner()

    def construct(self, x, y):
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        x = self.addn((x, y))
        x = self.inner_o3(x, y)
        return x


def test_ge_graph_mode_with_jit_level_o2():
    """
    Feature: GE with backend option.
    Description: Graph Mode backend="GE" with GE.
    Expectation: Run ok.
    """
    context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    net = NetOuter()
    net.set_jit_config(JitConfig(backend="GE"))
    output_o2 = net(inputs, inputs)
    expected = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], np.float32)
    np.allclose(output_o2.asnumpy(), expected, 1e-05, 1e-05)


def test_ge_graph_mode_without_jit_level():
    """
    Feature: GE with jit_level.
    Description: Graph Mode jit_level==None with GE.
    Expectation: Run ge without jit_level.
    """
    context.set_context(device_target="Ascend", mode=context.GRAPH_MODE)
    inputs = Tensor(np.ones((3, 3), np.float32))
    net = NetOuter()
    output = net(inputs, inputs)
    expected = np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], np.float32)
    np.allclose(output.asnumpy(), expected, 1e-05, 1e-05)


if __name__ == "__main__":
    test_ge_graph_mode_with_jit_level_o2()
    test_ge_graph_mode_without_jit_level()
