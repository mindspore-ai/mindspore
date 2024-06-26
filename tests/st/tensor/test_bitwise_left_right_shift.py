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


class Left(nn.Cell):
    def construct(self, x, other):
        return x.bitwise_left_shift(other)


class Right(nn.Cell):
    def construct(self, x, other):
        return x.bitwise_right_shift(other)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu'],
          level_mark='level2',
          card_mark='onecard',
          essential_mark='unessential')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_left_shift_right_shift_normal(mode):
    """
    Feature: tensor.bitwise_left_shift & tensor.bitwise_right_shift
    Description: Verify the result of the above tensor apis
    Expectation: success
    """
    ms.set_context(mode=mode)
    x = ms.Tensor(np.array([[1024, 5, 6]]), ms.int32)
    left = Left()
    right = Right()

    other_left = ms.Tensor(np.array([2]), ms.int32)
    other_right = ms.Tensor(np.array([1]), ms.int32)

    left_output = left(x, other_left)
    right_output = right(x, other_right)

    expected_left = np.array([4096, 20, 24], np.int32)
    expected_right = np.array([512, 2, 3], np.int32)

    assert left_output.dtype == x.dtype
    assert right_output.dtype == x.dtype
    assert np.allclose(left_output.asnumpy(), expected_left)
    assert np.allclose(right_output.asnumpy(), expected_right)
