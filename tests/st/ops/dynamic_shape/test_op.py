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

"""test op's dynamic shape rapidly"""

import pytest
from mindspore import Tensor, ops
import numpy as np
from .test_op_utils import TEST_OP


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sum():
    """
    Feature: test sum on cpu.
    Description: test all dynamic cases for sum.
    Expectation: the result match with expect
    """
    np_data1 = np.random.rand(2, 3, 4).astype(np.float32)
    in1 = Tensor(np_data1)
    np_data2 = np.random.rand(2, 3).astype(np.float32)
    in2 = Tensor(np_data2)

    reducesum = ops.ReduceSum(keep_dims=True)
    TEST_OP(reducesum, [[in1, [0]], [in2, [1]]], '', disable_yaml_check=True)
