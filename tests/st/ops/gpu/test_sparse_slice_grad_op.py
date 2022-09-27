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
"""smoke tests for SparseSliceGrad"""

import numpy as np
import pytest
from mindspore import Tensor, context
from mindspore.ops.operations._grad_ops import SparseSliceGrad


def generate_data(input_type="float32"):
    """
    generate data for sparse slice grad op test cases.
    """
    backprop_val_grad = Tensor(np.array([4, 2, 3]).astype(input_type))
    indices = Tensor(np.array([[0, 0], [0, 2], [1, 2], [1, 3], [2, 3], [2, 4]]).astype(np.int64))
    start = Tensor(np.array([0, 0]).astype(np.int64))
    new_indices = Tensor(np.array([[1, 2], [1, 3], [2, 3]]).astype(np.int64))
    data = backprop_val_grad, indices, start, new_indices
    return data


@pytest.mark.level1
@pytest.mark.parametrize('input_type', ["int8", "uint8", "int16", "uint16",
                                        "int32", "int64", "float16", "float32", "float64"])
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_sparse_tensor_dense_add(input_type):
    """
    Feature: Test sparse slice grad ops.
    Description: Test 2D sparse slice grad ops.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE)
    data = generate_data(input_type=input_type)
    net = SparseSliceGrad()
    out = net(data[0], data[1], data[2], data[3]).asnumpy()
    expected = np.array([0, 0, 4, 2, 3, 0]).astype(input_type)
    eps = 1e-6*np.array(np.ones_like(out))
    assert np.all(expected - out < eps)
