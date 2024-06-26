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
import mindspore
from mindspore.ops.function.array_func import arg_max_with_value_ as ArgMaxWithValue
from mindspore.ops.function.array_func import arg_min_with_value_ as ArgMinWithValue
from mindspore import Tensor
import numpy as np
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux'],
          level_mark='level0',
          card_mark='onecard',
          essential_mark='essential')
def test_pyboost_simple_infer():
    """
    Feature: test pyboost simple infer
    Description: test pyboost simple infer by pyboost
    Expectation: success
    """
    x = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
    for _ in range(1000):
        index_x, output_x = ArgMaxWithValue(x)
    assert output_x == 0.7
    assert index_x == 3

    y = Tensor(np.array([0.0, 0.4, 0.6, 0.7, 0.1]), mindspore.float32)
    for _ in range(1000):
        index_y, output_y = ArgMinWithValue(y)
    assert output_y == 0.0
    assert index_y == 0
