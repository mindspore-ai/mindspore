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
    def construct(self, x, axis):
        return x.min(axis)


@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_min(mode):
    """
    Feature: tensor.min
    Description: Verify the result of tensor.min
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = Tensor(np.arange(4).reshape((2, 2)).astype('float32'))
    axis = [1]
    net = Net()
    output = net(x, axis)
    expect_output = Tensor([0., 2.])
    assert np.allclose(output.asnumpy(), expect_output.asnumpy())
