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
        self.real = P.Real()

    def construct(self, input_ms):
        return self.real(input_ms)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_2D_complex64(context_mode):
    """
    Feature: aicpu ops Real.
    Description: test Real forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    real = np.random.randn(3, 4).astype(np.float32)
    img = np.random.randn(3, 4).astype(np.float32) * 1j
    input_np = real + img
    input_ms = Tensor(input_np, mstype.complex64)
    net = Net()
    output = net(input_ms)
    assert np.allclose(output.asnumpy(), real)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize("context_mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_net_3D_complex128(context_mode):
    """
    Feature: aicpu ops Real.
    Description: test Real forward.
    Expectation: expect correct result.
    """
    context.set_context(mode=context_mode, device_target="Ascend")
    real = np.random.randn(3, 4).astype(np.float64)
    img = np.random.randn(3, 4).astype(np.float64) * 1j
    input_np = real + img
    input_ms = Tensor(input_np, mstype.complex128)
    net = Net()
    output = net(input_ms)
    assert np.allclose(output.asnumpy(), real)
