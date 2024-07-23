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
import pytest
from tests.mark_utils import arg_mark
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y, z, beta, alpha):
        return x.addbmm(y, z, beta=beta, alpha=alpha)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_addbmm(mode):
    """
    Feature: tensor.addbmm
    Description: Verify the result of addbmm
    Expectation: success
    """
    ms.set_context(mode=mode)
    arr1 = np.arange(9).astype(np.float32).reshape((3, 3))
    arr2 = np.arange(24).astype(np.float32).reshape((2, 3, 4))
    arr3 = np.arange(24).astype(np.float32).reshape((2, 4, 3))
    x = Tensor(arr1)
    y = Tensor(arr2)
    z = Tensor(arr3)
    net = Net()
    output = net(x, y, z, 0.5, 2)
    expect_output = np.array([[1896.0000, 2016.5000, 2137.0000],
                              [2569.5000, 2754.0000, 2938.5000],
                              [3243.0000, 3491.5000, 3740.0000]])
    assert np.allclose(output.asnumpy(), expect_output)
