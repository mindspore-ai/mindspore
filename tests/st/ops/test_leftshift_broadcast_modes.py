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
"""LeftShift broadcast ST for multi-backend and multi-mode."""

from tests.mark_utils import arg_mark

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore.ops.operations.array_ops import LeftShift

ALL_INT_DTYPES = [
    np.int8, np.int16, np.int32, np.int64,
    np.uint8, np.uint16, np.uint32, np.uint64
]


class LeftShiftNet(ms.nn.Cell):
    """Net wrapper for LeftShift op."""

    def __init__(self):
        super().__init__()
        self.left_shift = LeftShift()

    def construct(self, x, y):
        return self.left_shift(x, y)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux'],
          level_mark='level0', card_mark='onecard', essential_mark='unessential', mem_peak=64.0)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', ALL_INT_DTYPES)
def test_leftshift_broadcast_multi_backend_multi_mode(mode, dtype):
    """
    Feature: LeftShift broadcast
    Description: Verify broadcast result in Graph O0 and PyNative on CPU/Ascend
    Expectation: Success
    """

    if mode == ms.GRAPH_MODE:
        ms.set_context(mode=mode, jit_level="O0")
    else:
        ms.set_context(mode=mode)


    x_np = np.array(
        [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]],
         [[13, 14], [15, 16]], [[17, 18], [19, 20]], [[21, 22], [23, 24]]],
        dtype=dtype
    )
    y_np = np.array([[0, 1], [2, 3]], dtype=dtype)
    expected = np.left_shift(x_np, y_np)

    out = LeftShiftNet()(Tensor(x_np), Tensor(y_np)).asnumpy()
    np.testing.assert_array_equal(out, expected)
