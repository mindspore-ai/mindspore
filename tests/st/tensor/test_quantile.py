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
        output = x.quantile(y)
        return output


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_quantile(mode):
    """
    Feature: tensor.quantile
    Description: Verify the result of tensor.quantile
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()
    x = Tensor(np.array([-0.7832, 0.8003, 0.8111]), ms.float32)
    q = Tensor(np.array([0.1, 0.7, 0.9]), ms.float32)
    output = net(x, q)
    expect_output = Tensor(np.asarray([-0.4665, 0.80462, 0.80894]), ms.float32)
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
