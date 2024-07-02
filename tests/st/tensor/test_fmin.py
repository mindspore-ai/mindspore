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
import mindspore as ms
import mindspore.nn as nn
from mindspore import Tensor


class Net(nn.Cell):
    def construct(self, x, y):
        return x.fmin(y)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_fmin(mode):
    """
    Feature: tensor.fmin
    Description: Verify the result of tensor.fmin
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.array([1.0, 5.0, 3.0]), ms.float32)
    y = Tensor(np.array([4.0, 2.0, 6.0]), ms.float32)
    net = Net()
    output = net(x, y)
    expect_output = Tensor(np.array([1.0, 2.0, 3.0]), ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
