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
    def construct(self, x, axis=None, keep_dims=False):
        return x.all(axis, keep_dims)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=1024.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_all(mode):
    """
    Feature: tensor.all
    Description: Verify the result of all
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([[True, False], [True, True]]))
    output = net(x, keep_dims=True) #mint.all(x, keepdim=True)
    assert np.allclose(output.asnumpy(), np.array([False]))
    assert np.allclose(output.shape, (1, 1))

    # case 2: Reduces a dimension along axis 0.
    output2 = net(x, axis=0)
    assert np.allclose(output2.asnumpy(), np.array([True, False]))
    assert np.allclose(output2.shape, (2,))
    # case 3: Reduces a dimension along axis 1.
    output3 = net(x, axis=1)
    assert np.allclose(output3.asnumpy(), np.array([False, True]))
    assert np.allclose(output3.shape, (2,))
