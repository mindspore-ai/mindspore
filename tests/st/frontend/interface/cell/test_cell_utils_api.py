# Copyright 2026 Huawei Technologies Co., Ltd
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
"""Smoke tests for Cell-related internal C++ pybind utilities."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          mem_peak=0.01)
def test_swap_tensor_impl_smoke():
    """
    Feature: _swap_tensor_impl.
    Description: Swap two tensors' underlying TensorImpl while preserving Python object ids.
    Expectation: Tensor contents are swapped and subsequent tensor ops work.
    """
    from mindspore._c_expression import _swap_tensor_impl

    ms.set_context(mode=ms.PYNATIVE_MODE)
    t1 = Tensor(np.ones((2, 2)), mstype.float32)
    t2 = Tensor(np.zeros((2, 2)), mstype.float32)
    id1, id2 = id(t1), id(t2)

    _swap_tensor_impl(t1, t2)

    assert id(t1) == id1
    assert id(t2) == id2
    np.testing.assert_array_equal(t1.asnumpy(), np.zeros((2, 2)))
    np.testing.assert_array_equal(t2.asnumpy(), np.ones((2, 2)))

    result = t1 + 1
    np.testing.assert_array_equal(result.asnumpy(), np.ones((2, 2)))


@arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
          level_mark='level0',
          card_mark='onecard',
          mem_peak=0.01)
def test_cell_parse_to_args_smoke():
    """
    Feature: _cell_parse_to_args.
    Description: Parse Cell.to args/kwargs into device, dtype and non_blocking tuple.
    Expectation: All three supported overloads parse correctly and unsupported forms are rejected.
    """
    from mindspore._c_expression import _cell_parse_to_args

    result = _cell_parse_to_args("Ascend", dtype=mstype.float16)
    assert isinstance(result, tuple)
    assert len(result) == 3
    assert result[0] == "Ascend"
    assert result[1] == mstype.float16
    assert result[2] is False

    result = _cell_parse_to_args("Ascend", dtype=mstype.float32, non_blocking=True)
    assert result[0] == "Ascend"
    assert result[1] == mstype.float32
    assert result[2] is True

    result = _cell_parse_to_args(mstype.float16)
    assert result[0] is None
    assert result[1] == mstype.float16
    assert result[2] is False

    other = Tensor(np.ones((2, 2)), mstype.float16)
    result = _cell_parse_to_args(other)
    assert result[0] == other.device
    assert result[1] == mstype.float16
    assert result[2] is False

    with pytest.raises(TypeError):
        _cell_parse_to_args(mstype.float16, mstype.int32)

    with pytest.raises(TypeError):
        _cell_parse_to_args("Ascend", copy=True)
