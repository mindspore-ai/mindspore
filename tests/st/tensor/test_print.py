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
from tests.mark_utils import arg_mark

import mindspore as ms
from mindspore import Tensor


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard')
def test_tensor_print():
    """
    Feature: Tensor print
    Description: Verify the result of tensor repr
    Expectation: success
    """
    x1 = Tensor(np.array([1, 2, 3]), ms.int32)
    x2 = Tensor(np.array([4, 5, 6]), ms.int32)
    y = (x1, x2)
    expect_str = "(Tensor(shape=[3], dtype=Int32, value= [1, 2, 3]), Tensor(shape=[3], dtype=Int32, value= [4, 5, 6]))"
    assert repr(y) == expect_str
