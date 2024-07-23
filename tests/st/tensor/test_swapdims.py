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
from tests.st.numpy_native.utils import to_tensor


class Net(nn.Cell):
    def construct(self, x, dim0, dim1):
        return x.swapdims(dim0, dim1)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_swapdims(mode):
    """
    Feature: Tensor.swapdims
    Description: Verify the result of swapdims
    Expectation: success
    """
    ms.set_context(mode=mode)
    lst = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    tensor_list = to_tensor(lst)
    net = Net()
    with pytest.raises(TypeError):
        tensor_list = net(tensor_list, 0, (1,))
    with pytest.raises(ValueError):
        tensor_list = net(tensor_list, 0, 3)
    assert net(tensor_list, 0, 1).shape == (3, 2)
