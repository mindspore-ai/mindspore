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
"""End-to-end tests for Cell.to / to_empty with traceable Tensor subclass."""
# pylint: disable=protected-access
import numpy as np

from mindspore import Tensor, Parameter
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from tests.st.frontend.interface.cell.conftest import MinimalTraceableTensor, assert_on_device


CELL_DTENSOR_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                             level_mark='level0',
                             card_mark='onecard',
                             mem_peak=0.01)


class TestCellToWithTraceableTensorSubclass:
    """Cell.to routes traceable Tensor subclass parameters through strategy A."""

    @CELL_DTENSOR_MARK
    def test_to_device_preserves_subclass_and_slot(self, simple_net):
        traceable_tensor = MinimalTraceableTensor(Tensor(np.ones((4, 4)), mstype.float32), payload="original")
        simple_net.w = Parameter(traceable_tensor)
        assert simple_net.w._payload_slot == "original"
        old_id = id(simple_net.w)

        simple_net.to(device="CPU")

        assert id(simple_net.w) == old_id
        assert isinstance(simple_net.w, MinimalTraceableTensor)
        assert simple_net.w._payload_slot == "original"
        assert_on_device(simple_net.w, "CPU")


class TestCellToEmptyWithTraceableTensorSubclass:
    """Cell.to_empty routes traceable Tensor subclass parameters through strategy A."""

    @CELL_DTENSOR_MARK
    def test_to_empty_on_traceable_tensor_param(self, simple_net):
        traceable_tensor = MinimalTraceableTensor(Tensor(np.ones((4, 4)), mstype.float32), payload="p")
        simple_net.w = Parameter(traceable_tensor)
        old_id = id(simple_net.w)

        simple_net.to_empty(device="CPU")

        assert id(simple_net.w) == old_id
        assert isinstance(simple_net.w, MinimalTraceableTensor)
        assert_on_device(simple_net.w, "CPU")
