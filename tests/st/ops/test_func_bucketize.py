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
from tests.mark_utils import arg_mark
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops


class Net(nn.Cell):
    def construct(self, input_x, boundaries, right):
        return ops.bucketize(input_x, boundaries, right=right)


@arg_mark(plat_marks=['platform_ascend', 'platform_gpu', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2', card_mark='onecard', essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test ops.bucketize
    Description: verify the result of bucketize
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()

    values = Tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]])
    boundaries = [1, 2, 3, 4, 5, 6]

    expected_result = np.array([[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]], np.int32)
    output = net(values, boundaries, right=False)
    assert np.allclose(output.asnumpy(), expected_result)

    expected_result = np.array([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], np.int32)
    output = net(values, boundaries, right=True)
    assert np.allclose(output.asnumpy(), expected_result)
    