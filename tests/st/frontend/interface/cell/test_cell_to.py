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
"""Tests for Cell.to()."""
import numpy as np
import pytest

from mindspore import Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from tests.st.frontend.interface.cell.conftest import assert_on_device, allocated_bytes


CELL_TO_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                        level_mark='level0',
                        card_mark='onecard',
                        mem_peak=0.01)


class TestToOverload1:
    """to(device=..., dtype=..., non_blocking=...)"""

    @CELL_TO_MARK
    def test_to_device_only(self, simple_net):
        simple_net.to(device="CPU")

        assert_on_device(simple_net.w, "CPU")
        assert_on_device(simple_net.buf, "CPU")

    @CELL_TO_MARK
    def test_to_dtype_only_via_kwarg(self, simple_net):
        simple_net.to(dtype=mstype.float16)

        assert simple_net.w.dtype == mstype.float16

    @CELL_TO_MARK
    def test_to_device_and_dtype(self, simple_net):
        simple_net.to(device="CPU", dtype=mstype.float16)

        assert_on_device(simple_net.w, "CPU")
        assert simple_net.w.dtype == mstype.float16

    @CELL_TO_MARK
    def test_integral_buffer_dtype_unchanged(self, simple_net):
        original_buf_dtype = simple_net.buf.dtype

        simple_net.to(dtype=mstype.float16)

        assert simple_net.w.dtype == mstype.float16
        assert simple_net.buf.dtype == original_buf_dtype


class TestToOverload2:
    """to(dtype)"""

    @CELL_TO_MARK
    def test_to_dtype_positional(self, simple_net):
        simple_net.to(mstype.float16)

        assert simple_net.w.dtype == mstype.float16


class TestToOverload3:
    """to(tensor)"""

    @CELL_TO_MARK
    def test_to_tensor(self, simple_net):
        ref = Tensor(np.zeros((1,)), mstype.float16)

        simple_net.to(ref)

        assert simple_net.w.dtype == ref.dtype


class TestToMeta:
    """Meta device conversions."""

    @CELL_TO_MARK
    def test_real_to_meta_works(self, simple_net):
        simple_net.to(device="meta")

        assert simple_net.w.is_meta
        assert allocated_bytes(simple_net.w) == 0

    @CELL_TO_MARK
    def test_meta_to_real_raises_with_hint(self, simple_net):
        simple_net.to(device="meta")

        with pytest.raises(RuntimeError, match="Please use Cell.to_empty"):
            simple_net.to(device="CPU")


class TestToErrors:
    """Error paths for Cell.to."""

    @CELL_TO_MARK
    def test_invalid_dtype_non_floating(self, simple_net):
        with pytest.raises(TypeError, match="only accepts floating point or complex"):
            simple_net.to(dtype=mstype.int32)

    @CELL_TO_MARK
    def test_dtype_dtype_rejected_by_parser(self, simple_net):
        with pytest.raises(TypeError):
            simple_net.to(mstype.float16, mstype.float32)

    @CELL_TO_MARK
    def test_copy_kwarg_rejected_by_parser(self, simple_net):
        with pytest.raises(TypeError):
            simple_net.to(device="CPU", copy=True)


class TestToReturnsAndInPlace:
    """Return value and object identity behavior."""

    @CELL_TO_MARK
    def test_returns_self(self, simple_net):
        result = simple_net.to(device="CPU")

        assert result is simple_net

    @CELL_TO_MARK
    def test_parameter_id_preserved(self, simple_net):
        old_id = id(simple_net.w)

        simple_net.to(device="CPU")

        assert id(simple_net.w) == old_id
