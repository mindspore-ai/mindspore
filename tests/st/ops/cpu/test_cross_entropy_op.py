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

import mindspore.common.dtype as mstype
from mindspore.common.tensor import Tensor
import mindspore.ops as ops


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cross_entropy_ops_input_float64():
    """
    Feature: Test abnormal input dtype of CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional.
    Expectation: Success.
    """
    input_data = Tensor(np.random.randn(3, 5).astype(np.float64))
    target_data = Tensor(np.random.randint(0, 5, (3,)), mstype.int32)
    weight_data = Tensor(np.random.randn(5,), mstype.float32)
    ops.cross_entropy(input_data, target_data, weight_data)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cross_entropy_ops_weight_float64():
    """
    Feature: Test abnormal weight dtype of CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional.
    Expectation: Success.
    """
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.random.randint(0, 5, (3,)), mstype.int32)
    weight_data = Tensor(np.random.randn(5,), mstype.float64)
    ops.cross_entropy(input_data, target_data, weight_data)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_cross_entropy_ops_target_int64():
    """
    Feature: Test abnormal target dtype of CrossEntropyLoss.
    Description: Test CrossEntropyLoss functional.
    Expectation: Success.
    """
    input_data = Tensor(np.random.randn(3, 5).astype(np.float32))
    target_data = Tensor(np.random.randint(0, 5, (3,)), mstype.int64)
    weight_data = Tensor(np.random.randn(5,), mstype.float32)
    ops.cross_entropy(input_data, target_data, weight_data)
