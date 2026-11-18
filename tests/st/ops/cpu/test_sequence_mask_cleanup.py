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
"""Sequence-mask wrapper removal preserves independent RNN and shared ops behavior."""
import numpy as np
import pytest
import mindspore as ms
from mindspore import nn, ops
from mindspore.nn.layer import rnns
from tests.mark_utils import arg_mark


class RNNMask(nn.Cell):
    def construct(self, lengths):
        return rnns.sequence_mask(lengths, 4)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [ms.int32, ms.int64])
def test_retained_rnn_mask(mode, dtype):
    """
    Feature: Independent RNN sequence masking.
    Description: Evaluate negative, zero, partial, full and overlong lengths in both execution modes.
    Expectation: The unchanged RNN helper returns the expected int32 mask.
    """
    previous = ms.get_context('mode')
    try:
        ms.set_context(mode=mode)
        lengths = ms.Tensor([-1, 0, 2, 4, 6], dtype)
        result = RNNMask()(lengths)
        expected = (np.arange(4) < np.array([-1, 0, 2, 4, 6])[:, None]).astype(np.int32)
        np.testing.assert_array_equal(result.asnumpy(), expected)
        assert result.dtype == ms.int32
    finally:
        ms.set_context(mode=previous)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
def test_retained_shared_array_operations():
    """
    Feature: Shared array operations after cache removal.
    Description: Run repeat_elements and max using retained helpers and primitive classes.
    Expectation: Repetition and maximum indices/values match NumPy.
    """
    data = ms.Tensor([[1., 4.], [3., 2.]], ms.float32)
    np.testing.assert_array_equal(ops.repeat_elements(data, 2, 1).asnumpy(),
                                  np.repeat(data.asnumpy(), 2, axis=1))
    values, indices = ops.max(data, axis=1)
    np.testing.assert_array_equal(indices.asnumpy(), [1, 0])
    np.testing.assert_array_equal(values.asnumpy(), [4., 3.])
