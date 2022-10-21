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
"""smoke tests for SparseSlice"""

import numpy as np
import pytest
from mindspore import Tensor, context
from mindspore.ops.operations.sparse_ops import SparseSlice


def generate_data(input_type="float32"):
    """
    generate data for sparse slice op test cases.
    """
    indices = Tensor(np.array([[0, 1], [1, 2], [1, 3], [2, 2]])).astype(np.int64)
    values = Tensor(np.array([1, 2, 3, 4])).astype(input_type)
    shape = Tensor(np.array([3, 4])).astype(np.int64)
    start = Tensor(np.array([0, 1])).astype(np.int64)
    size = Tensor(np.array([2, 3])).astype(np.int64)
    data = indices, values, shape, start, size
    return data


@pytest.mark.level1
@pytest.mark.parametrize('input_type', ["int8", "uint8", "int16", "uint16",
                                        "int32", "int64", "float16", "float32", "float64"])
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_slice(input_type):
    """
    Feature: Test sparse slice ops.
    Description: Test 2D sparse slice ops.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data = generate_data(input_type=input_type)
    net = SparseSlice()
    out = net(data[0], data[1], data[2], data[3], data[4])
    out_indices = out[0].asnumpy()
    out_values = out[1].asnumpy()
    out_size = out[2].asnumpy()
    expected_indices = np.array([[0, 0], [1, 1], [1, 2]]).astype(np.int64)
    expected_values = np.array([1, 2, 3]).astype(input_type)
    expected_size = np.array([2, 3]).astype(np.int64)
    eps_indices = 1e-6 * np.array(np.ones_like(out_indices))
    eps_values = 1e-6 * np.array(np.ones_like(out_values))
    eps_size = 1e-6 * np.array(np.ones_like(out_size))

    assert np.all(expected_indices - out_indices < eps_indices)
    assert np.all(expected_values - out_values < eps_values)
    assert np.all(expected_size - out_size < eps_size)
