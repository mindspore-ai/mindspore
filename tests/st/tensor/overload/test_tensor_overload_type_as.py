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

import mindspore as ms
import mindspore.common.dtype as mstype
import mindspore.nn as nn

from tests.mark_utils import arg_mark


class Net(nn.Cell):

    def construct(self, x, other):
        return x.type_as(other)


@arg_mark(plat_marks=[
    'cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend',
    'platform_ascend910b'
],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='essential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_type_as(mode):
    """
    Feature: tensor.type_as
    Description: Verify the result of type_as
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    x = ms.Tensor([[1, 2, 3], [2, 3, 4]], mstype.float32)
    y = ms.Tensor([1, 1, 1, 1, 1, 1], mstype.int32)
    net = Net()
    output = net(x, y)
    assert output.asnumpy().dtype == "int32"
