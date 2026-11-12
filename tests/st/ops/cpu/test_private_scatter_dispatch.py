# Copyright 2021 Huawei Technologies Co., Ltd
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
"""Public API and stateful scatter dispatch regression tests."""
import numpy as np
import pytest

from mindspore import Parameter, Tensor, ops
from mindspore.ops import operations
from mindspore.ops.operations import _scatter_ops
from mindspore.ops._vmap.vmap_base import vmap_rules_getters
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize("name,expected", [
    ("Div", 2), ("Max", 12), ("Min", 2), ("Mul", 72)])
def test_private_scatter_dispatch(name, expected):
    """
    Feature: Private stateful scatter implementation.
    Description: Update a Parameter at a repeated index via the private primitive.
    Expectation: Public classes are absent; dispatch, vmap registration and writeback survive.
    """
    primitive = "Scatter" + name
    assert not hasattr(ops, primitive)
    assert not hasattr(operations, primitive)
    private = getattr(_scatter_ops, primitive)()
    assert private.name == primitive
    get_rule = vmap_rules_getters.get(private, None)
    assert get_rule is not None
    value = Parameter(Tensor(np.array([12, 5], np.float32)))
    with pytest.raises(ValueError, match="execution order"):
        get_rule(private, 2)((value, None), (Tensor([0, 0]), 0), (Tensor([2, 3]), 0), None)
    out = private(
        value, Tensor(np.array([0, 0], np.int32)), Tensor(np.array([2, 3], np.float32)))
    np.testing.assert_array_equal(out.asnumpy(), [expected, 5])
    np.testing.assert_array_equal(value.asnumpy(), [expected, 5])
