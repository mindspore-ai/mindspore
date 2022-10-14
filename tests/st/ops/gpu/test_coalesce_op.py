# Copyright 2022 Huawei Technologies Co., Ltd

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.common import dtype as mstype
from mindspore.ops.operations import array_ops as aps


context.set_context(mode=context.GRAPH_MODE, device_target="GPU")


class NetCoalesce(nn.Cell):
    def __init__(self):
        super(NetCoalesce, self).__init__()
        self.coalesce = aps.Coalesce()

    def construct(self, x, y, z):
        return self.coalesce(x, y, z)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coalesce_fp32():
    """
    Feature: Coalesce function.
    Description:  The Tensor of int64, float32 and int64.
    Expectation: Returns the coalesced sparse tensor of the input.
    """
    coalesce = NetCoalesce()
    x_indices = Tensor([[1, 2, 3, 3, 2], [2, 2, 2, 2, 2]], dtype=mstype.int64)
    x_values = Tensor([1, 2, 3, 4, 5], dtype=mstype.float32)
    x_shape = Tensor([5, 5], dtype=mstype.int64)
    y_indices, y_values, y_shape = coalesce(x_indices, x_values, x_shape)
    expect_indices = np.array([[1, 2, 3], [2, 2, 2]]).astype(np.int64)
    expect_values = np.array([1, 7, 7]).astype(np.float32)
    expect_shape = np.array([5, 5]).astype(np.int64)
    assert np.array_equal(y_indices, expect_indices)
    assert np.array_equal(y_values, expect_values)
    assert np.array_equal(y_shape, expect_shape)


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coalesce_fp16():
    """
    Feature: Coalesce function.
    Description:  The Tensor of int64, float16 and int64.
    Expectation: Returns the coalesced sparse tensor of the input.
    """
    coalesce = NetCoalesce()
    x_indices = Tensor([[1, 2, 3, 3, 2], [2, 2, 2, 2, 2]], dtype=mstype.int64)
    x_values = Tensor([1, 2, 3, 4, 5], dtype=mstype.float16)
    x_shape = Tensor([5, 5], dtype=mstype.int64)
    y_indices, y_values, y_shape = coalesce(x_indices, x_values, x_shape)
    expect_indices = np.array([[1, 2, 3], [2, 2, 2]]).astype(np.int64)
    expect_values = np.array([1, 7, 7]).astype(np.float16)
    expect_shape = np.array([5, 5]).astype(np.int64)
    assert np.array_equal(y_indices, expect_indices)
    assert np.array_equal(y_values, expect_values)
    assert np.array_equal(y_shape, expect_shape)


@pytest.mark.level0gi
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_coalesce_fp64():
    """
    Feature: Coalesce function.
    Description:  The Tensor of int64, float64 and int64.
    Expectation: Returns the coalesced sparse tensor of the input.
    """
    coalesce = NetCoalesce()
    x_indices = Tensor([[1, 3, 3], [2, 3, 3]], dtype=mstype.int64)
    x_values = Tensor([1, 1, 1], dtype=mstype.float64)
    x_shape = Tensor([5, 5], dtype=mstype.int64)
    y_indices, y_values, y_shape = coalesce(x_indices, x_values, x_shape)
    expect_indices = np.array([[1, 3], [2, 3]]).astype(np.int64)
    expect_values = np.array([1, 2]).astype(np.float64)
    expect_shape = np.array([5, 5]).astype(np.int64)
    assert np.array_equal(y_indices, expect_indices)
    assert np.array_equal(y_values, expect_values)
    assert np.array_equal(y_shape, expect_shape)
