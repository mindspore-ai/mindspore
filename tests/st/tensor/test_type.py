# Copyright 2023 Huawei Technologies Co., Ltd
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
    def __init__(self, dtype=None):
        super(Net, self).__init__()
        self.dtype = dtype

    def construct(self, x):
        if self.dtype is None:
            return x.type()
        return x.type(self.dtype)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level1',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_type(mode):
    """
    Feature: tensor.type
    Description: Verify the result of output
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor([[-1.1, 0.3, 3.6], [10.4, 3.5, -3.2]], dtype=ms.float32)
    net = Net()
    output = net(x)
    expect_output1 = "Float32"
    assert output == expect_output1
    net = Net(dtype=ms.int32)
    output = net(x)
    expect_output2 = [[-1, 0, 3], [10, 3, -3]]
    assert np.allclose(output.asnumpy(), expect_output2)
