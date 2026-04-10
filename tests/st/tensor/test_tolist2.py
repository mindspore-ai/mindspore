# Copyright 2025 Huawei Technologies Co., Ltd
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
from mindspore import nn, Tensor


class Net(nn.Cell):
    def construct(self, x):
        return x.tolist()


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_tensor_tolist(mode):
    """
    Feature: tensor.tolist
    Description: Verify the result of tolist
    Expectation: success
    """
    ms.set_context(mode=mode)
    net = Net()

    x = Tensor([])
    expect_output = []
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor([1])
    expect_output = [1]
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor([1, 2])
    expect_output = [1, 2]
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor([[1], [4]])
    expect_output = [[1], [4]]
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor([[1, 2, 3], [4, 5, 6]])
    expect_output = [[1, 2, 3], [4, 5, 6]]
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor([[[1], [2]], [[4], [5]]])
    expect_output = [[[1], [2]], [[4], [5]]]
    output = net(x)
    assert np.allclose(output, expect_output)

    x = Tensor(np.random.randn(2, 0), ms.float32)
    expect_output = [[], []]
    output = net(x)
    assert output == expect_output
