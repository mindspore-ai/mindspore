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
    def construct(self, x, dim=None, keepdim=False):
        return x.argmin(dim=dim, keepdim=keepdim)


class Net2(nn.Cell):
    def construct(self, x, axis=None, keepdims=False):
        return x.argmin(axis=axis, keepdims=keepdims)

def net_test(x):
    net = Net()
    output = net(x, dim=-1)
    assert output.shape == (3,)
    assert np.allclose(output.asnumpy(), np.array([0, 1, 2]))
    output2 = net(x, dim=-1, keepdim=True)
    assert output2.shape == (3, 1)
    assert np.allclose(output2.asnumpy(), np.array([[0], [1], [2]]))

def net2_test(x):
    net = Net2()
    output = net(x, axis=-1)
    assert output.shape == (3,)
    assert np.allclose(output.asnumpy(), np.array([0, 1, 2]))
    output2 = net(x, axis=-1, keepdims=True)
    assert output2.shape == (3, 1)
    assert np.allclose(output2.asnumpy(), np.array([[0], [1], [2]]))

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential', mem_peak=4.0)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_tensor_argmin(mode):
    """
    Feature: tensor.argmin
    Description: Verify the result of argmin
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([[1, 20, 5], [67, 8, 9], [130, 24, 15]]).astype(np.float32))
    net2_test(x)
    if ms.get_context('device_target') == 'Ascend':
        net_test(x)
