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
import pytest
from tests.mark_utils import arg_mark
from mindspore import context
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops

context.set_context(mode=context.PYNATIVE_MODE)


class Net(nn.Cell):
    def __init__(self):
        super().__init__()
        self.i = 1

    def construct(self, x):
        return bool(x)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level0',
          card_mark='onecard')
def test_tensor_bool():
    """
    Feature: bool(tensor)
    Description: Verify the result of bool
    Expectation: success
    """
    net = Net()

    x = ms.Tensor([0], ms.int32)
    assert ~net(x)
    y = ops.abs(x)
    assert ~net(y)

    x = ms.Tensor([1], ms.int32)
    assert net(x)
    y = ops.abs(x)
    assert net(y)

    x = ms.Tensor([])
    with pytest.raises(ValueError):
        net(x)

    x = ms.Tensor([0, 1], ms.int32)
    y = ops.abs(x)
    with pytest.raises(ValueError):
        net(x)
    with pytest.raises(ValueError):
        net(y)
