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
"""smoke tests for SparseSplit"""
import pytest
import numpy as np
import mindspore.common.dtype as mstype
from mindspore import Tensor, context
from mindspore.ops.operations.sparse_ops import SparseSplit


@pytest.mark.level0
@pytest.mark.platform_x86_gpu_training
@pytest.mark.env_onecard
def test_ops_sparse_split_vs_tf_output_int64():
    """
    Feature: Test function sparse_split.
    Description: Test sparse_split compared with tf.sparse_split.
    Expectation: Success.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target='GPU')
    indices = Tensor([[0, 0], [1, 1]], mstype.int64)
    values = Tensor([1, 2], mstype.int64)
    dense_shape = (2, 2)
    split_dim = 1
    num_split = 2
    sparsesplit = SparseSplit(num_split=num_split)
    output1 = sparsesplit(Tensor(split_dim, mstype.int64), indices, values, Tensor(dense_shape, mstype.int64))
    tf_indices_0 = [0, 0]
    tf_indices_1 = [1, 0]
    tf_values_0 = 1
    tf_values_1 = 2
    assert np.allclose(output1[0].asnumpy(), tf_indices_0)
    assert np.allclose(output1[1].asnumpy(), tf_indices_1)
    assert np.allclose(output1[2].asnumpy(), tf_values_0)
    assert np.allclose(output1[3].asnumpy(), tf_values_1)
