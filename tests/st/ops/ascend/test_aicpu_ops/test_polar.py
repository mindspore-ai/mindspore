# Copyright 2024 Huawei Technologies Co., Ltd
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
import pytest
import numpy as np

import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype


class Net(nn.Cell):
    def __init__(self):
        super(Net, self).__init__()
        self.polar = P.Polar()

    def construct(self, abs_ms, angle_ms):
        return self.polar(abs_ms, angle_ms)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_2D_float32(context_mode):
    """
    Feature: aicpu ops Polar.
    Description: test Polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    abs_np = np.random.randn(3, 4).astype(np.float32)
    angle_np = np.random.randn(3, 4).astype(np.float32)
    net = Net()
    abs_ms, angle_ms = Tensor(abs_np, mstype.float32), Tensor(angle_np, mstype.float32)
    output = net(abs_ms, angle_ms)
    expect = abs_np * (np.cos(angle_np)) + 1j * abs_np * (np.sin(angle_np))
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_3D_float64(context_mode):
    """
    Feature: aicpu ops Polar.
    Description: test Polar forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    abs_np = np.random.randn(3, 4, 5).astype(np.float64)
    angle_np = np.random.randn(3, 4, 5).astype(np.float64)
    net = Net()
    abs_ms, angle_ms = Tensor(abs_np, mstype.float64), Tensor(angle_np, mstype.float64)
    output = net(abs_ms, angle_ms)
    expect = abs_np * (np.cos(angle_np)) + 1j * abs_np * (np.sin(angle_np))
    assert np.allclose(output.asnumpy(), expect)
