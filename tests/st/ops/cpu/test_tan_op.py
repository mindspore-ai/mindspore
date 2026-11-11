# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Test tan op."""
from mindspore.ops.operations import _inner_ops as _private_math
import mindspore.nn as nn
from mindspore import Tensor
from mindspore import context

import numpy as np

from tests.mark_utils import arg_mark

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class NetTan(nn.Cell):
    def __init__(self):
        super().__init__()
        self.tan = _private_math.Tan()

    def construct(self, x):
        return self.tan(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
def test_tan():
    np_array = np.array([-1, -0.5, 0, 0.5, 1]).astype('float32')
    input_x = Tensor(np_array)
    net = NetTan()
    output = net(input_x)
    print(output)
    expect = np.tan(np_array)
    assert np.allclose(output.asnumpy(), expect)
