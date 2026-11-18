# Copyright 2020-2024 Huawei Technologies Co., Ltd
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
"""Regression tests for private indexed arithmetic primitives."""
import numpy as np
import pytest
from mindspore import ops, Tensor
from mindspore.ops import operations
from mindspore.ops.operations import _inplace_ops
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("suffix", ["Add", "Sub"])
def test_private_inplace_dispatch(suffix):
    """
    Feature: Retained indexed arithmetic functions.
    Description: Run eager and batched updates with repeated row indices.
    Expectation: Results match NumPy and neither public namespace exposes the primitive.
    """
    name = "Inplace" + suffix
    assert not hasattr(ops, name)
    assert not hasattr(operations, name)
    assert getattr(_inplace_ops, name)((0, 0)).name == name
    function = getattr(_inplace_ops, name)((0, 0))
    x = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
    updates = np.arange(1, 9, dtype=np.float32).reshape(2, 2, 2)
    expected = x.copy()
    op = np.add if suffix == "Add" else np.subtract
    for i in range(2):
        op.at(expected[i], [0, 0], updates[i])

    def apply_rows(row, values):
        return function(row, values)

    np.testing.assert_array_equal(apply_rows(Tensor(x[0]), Tensor(updates[0])).asnumpy(), expected[0])
    result = ops.vmap(apply_rows, in_axes=(0, 0))(Tensor(x), Tensor(updates))
    np.testing.assert_array_equal(result.asnumpy(), expected)
