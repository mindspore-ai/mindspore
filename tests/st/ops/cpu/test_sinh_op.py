# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
from mindspore import Tensor
from mindspore import context
from mindspore.ops import operations as P
from mindspore.common import dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetSinh(nn.Cell):
    def __init__(self):
        super(NetSinh, self).__init__()
        self.sinh = P.Sinh()

    def construct(self, x):
        return self.sinh(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sinh():
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = NetSinh()
    output = net(input_x)
    print(output)
    expect = np.sinh(np_array)
    assert np.allclose(output.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_sinh_tensor_api_modes(mode):
    """
    Feature: Test sinh tensor api.
    Description: Test sinh tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor([0.62, 0.28, 0.43, 0.62], mstype.float32)
    output = x.sinh()
    expected = np.array([0.6604918, 0.28367308, 0.44337422, 0.6604918], np.float32)
    np.testing.assert_array_almost_equal(output.asnumpy(), expected, decimal=4)
