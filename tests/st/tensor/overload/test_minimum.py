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
import numpy as np
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, other):
        return x.minimum(other)


@arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
    level_mark='level1',
    card_mark='onecard',
    essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_minimum(mode):
    """
    Feature: tensor.minimum
    Description: Verify the result of tensor.minimum
    Expectation: success
    """
    ms.set_context(mode=mode, jit_config={"jit_level": "O0"})
    # test1: same dtype
    x = Tensor(np.array([1.0, 5.0, 3.0]), ms.float32)
    other = Tensor(np.array([4.0, 2.0, 6.0]), ms.float32)
    net = Net()
    output_x = net(x, other)
    expect_x = Tensor(np.array([1, 2, 3]), ms.float32)
    assert np.allclose(output_x.asnumpy(), expect_x.asnumpy())

    # test2: different dtype
    x = Tensor(np.array([7.0, 5.0, 3.0]), ms.int32)
    other = Tensor(np.array([4.0, 2.0, 11.0]), ms.float32)
    net = Net()
    output_x = net(x, other)
    assert output_x.dtype == ms.float32
