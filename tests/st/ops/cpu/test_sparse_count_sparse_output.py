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
from tests.mark_utils import arg_mark

import numpy as np
import pytest
import mindspore.context as context
import mindspore.nn as nn
from mindspore import Tensor
from mindspore.ops.operations import sparse_ops as P
import mindspore.common.dtype as mstype

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")


class Net(nn.Cell):
    def __init__(self, binary_output, minlength, maxlength):
        super(Net, self).__init__()
        self.sparse_count_sparse_output = P.SparseCountSparseOutput(binary_output=binary_output, \
                                            minlength=minlength, maxlength=maxlength)

    def construct(self, indices, values, dense_shape, weights):
        out = self.sparse_count_sparse_output(indices, values, dense_shape, weights)
        return out


def compare_results(result, expect):
    indices_result = result[0].asnumpy()
    values_result = result[1].asnumpy()
    dense_shape_result = result[2].asnumpy()
    assert indices_result.shape == expect[0].asnumpy().shape
    assert np.allclose(indices_result, expect[0].asnumpy())
    assert values_result.dtype == expect[1].asnumpy().dtype
    assert values_result.shape == expect[1].asnumpy().shape
    assert np.allclose(values_result, expect[1].asnumpy())
    assert dense_shape_result.shape == expect[2].asnumpy().shape
    assert np.allclose(dense_shape_result, expect[2].asnumpy())


def sparse_count_sparse_output_valuecheck(v_type, w_type):
    indices = Tensor([[1, 2], [3, 4], [2, 1], [2, 2], [1, 0], [3, 3], [2, 0], [2, 2]], \
                    dtype=mstype.int64)
    values = Tensor([0, 2, 8, 8, 1, 2, 3, 7], dtype=v_type)
    dense_shape = Tensor([5, 5], dtype=mstype.int64)
    weights = Tensor([2, 5, 1, 0, 4, 2, 2, 2], dtype=w_type)
    sparse_count_sparse_output = Net(binary_output=False, minlength=-1, maxlength=-1)
    op_output = sparse_count_sparse_output(indices, values, dense_shape, weights)
    expect_indices = Tensor([[1, 0], [1, 1], [2, 3], [2, 7], [2, 8], [3, 2]], dtype=mstype.int64)
    expect_values = Tensor([2, 4, 2, 2, 1, 7], dtype=w_type)
    expect_shape = Tensor([5, 9], dtype=mstype.int64)
    expected_output = (expect_indices, expect_values, expect_shape)
    compare_results(op_output, expected_output)

    #Test with float values for weights
    indices = Tensor([[1, 2], [3, 4], [2, 1], [2, 2], [1, 0], [3, 3], [2, 0], [2, 2]], \
                    dtype=mstype.int64)
    values = Tensor([0, 2, 8, 8, 1, 2, 3, 7], dtype=v_type)
    dense_shape = Tensor([5, 5], dtype=mstype.int64)
    weights = Tensor([2.2, 5.7, 1, 0, 4.6, 2, 2, 2], dtype=mstype.float32)
    sparse_count_sparse_output = Net(binary_output=False, minlength=-1, maxlength=-1)
    op_output = sparse_count_sparse_output(indices, values, dense_shape, weights)
    expect_indices = Tensor([[1, 0], [1, 1], [2, 3], [2, 7], [2, 8], [3, 2]], dtype=mstype.int64)
    expect_values = Tensor([2.2, 4.6, 2, 2, 1, 7.7], dtype=mstype.float32)
    expect_shape = Tensor([5, 9], dtype=mstype.int64)
    expected_output = (expect_indices, expect_values, expect_shape)
    compare_results(op_output, expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparsecountsparseoutput_checkvalue_difftypes():
    """
    Feature: SparseCountSparseOutput cpu op
    Description: Test output for the op
    Expectation: Output matching expected values
    """
    values_types = (mstype.int32, mstype.int64)
    weights_types = (mstype.int32, mstype.int64)
    for v_type in values_types:
        for w_type in weights_types:
            sparse_count_sparse_output_valuecheck(v_type, w_type)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard', essential_mark='essential')
def test_sparsecountsparseoutput_checkvalue_maxvalue():
    """
    Feature: SparseCountSparseOutput cpu op
    Description: Test output for the op with maxvalue input
    Expectation: Output matching expected values
    """
    indices = Tensor([[1, 2], [3, 4], [2, 1], [2, 2], [1, 0], [3, 3], [2, 0], [2, 2]], \
                    dtype=mstype.int64)
    values = Tensor([0, 2, 8, 8, 1, 2, 3, 7], dtype=mstype.int64)
    dense_shape = Tensor([5, 5], dtype=mstype.int64)
    weights = Tensor([2, 5, 1, 0, 4, 2, 2, 2], dtype=mstype.int64)
    sparse_count_sparse_output = Net(binary_output=False, minlength=-1, maxlength=4)
    op_output = sparse_count_sparse_output(indices, values, dense_shape, weights)
    expect_indices = Tensor([[1, 0], [1, 1], [2, 3], [3, 2]], dtype=mstype.int64)
    expect_values = Tensor([2, 4, 2, 7], dtype=mstype.int64)
    expect_shape = Tensor([5, 4], dtype=mstype.int64)
    expected_output = (expect_indices, expect_values, expect_shape)
    compare_results(op_output, expected_output)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard', essential_mark='unessential')
def test_sparsecountsparseoutput_value_type_error():
    """
    Feature: SparseCountSparseOutput cpu op
    Description: Test output for unsupported type or value bounds
    Expectation: Raises corresponding errors
    """
    #Number of weights is not equal to number of values
    with pytest.raises(RuntimeError):
        indices = Tensor([[1, 2], [3, 1], [2, 2], [2, 1]], dtype=mstype.int64)
        values = Tensor([0, 2, 8, 8], dtype=mstype.int64)
        dense_shape = Tensor([4, 4], dtype=mstype.int64)
        weights = Tensor([1, 2, 1], dtype=mstype.int64)
        sparse_count_sparse_output = Net(binary_output=False, minlength=-1, maxlength=-1)
        sparse_count_sparse_output(indices, values, dense_shape, weights)

    #Indexes are not in bound of dense shape
    with pytest.raises(RuntimeError):
        indices = Tensor([[1, 2], [8, 6], [2, 2], [2, 1]], dtype=mstype.int64)
        values = Tensor([0, 2, 8, 8], dtype=mstype.int64)
        dense_shape = Tensor([4, 3], dtype=mstype.int64)
        weights = Tensor([1, 2, 1, 0], dtype=mstype.int64)
        sparse_count_sparse_output = Net(binary_output=False, minlength=-1, maxlength=-1)
        sparse_count_sparse_output(indices, values, dense_shape, weights)
