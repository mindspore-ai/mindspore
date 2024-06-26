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

import pytest
from tests.mark_utils import arg_mark

import mindspore as ms
import mindspore.nn as nn


class Net(nn.Cell):
    def construct(self, x):
        return x.is_complex()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_is_complex(mode):
    """
    Feature: tensor.is_complex
    Description: Verify the result of is_complex
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    a = ms.Tensor([complex(2, 3), complex(1, 3), complex(2.2, 3)], ms.complex64)
    b = ms.Tensor(complex(2, 3), ms.complex128)
    c = ms.Tensor([1, 2, 3], ms.float32)
    out1 = net(a)
    out2 = net(b)
    out3 = net(c)
    assert out1
    assert out2
    assert not out3
