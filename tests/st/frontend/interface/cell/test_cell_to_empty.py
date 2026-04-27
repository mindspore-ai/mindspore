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
"""Tests for Cell.to_empty()."""
import numpy as np
import pytest

from mindspore import Tensor, Parameter, nn
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark
from tests.st.frontend.interface.cell.conftest import assert_on_device, allocated_bytes


CELL_TO_EMPTY_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                              level_mark='level0',
                              card_mark='onecard',
                              mem_peak=0.01)


class TestToEmptyDevice:
    """Device behavior for Cell.to_empty."""

    @CELL_TO_EMPTY_MARK
    def test_to_empty_cpu(self, simple_net):
        simple_net.to_empty(device="CPU")

        for param in simple_net.get_parameters():
            assert_on_device(param, "CPU")
            assert allocated_bytes(param) > 0

    @CELL_TO_EMPTY_MARK
    def test_to_empty_meta_zero_memory(self, simple_net):
        simple_net.to_empty(device="meta")

        for param in simple_net.get_parameters():
            assert param.is_meta
            assert allocated_bytes(param) == 0


class TestToEmptyMetaToReal:
    """Main use case: meta init to real device allocation."""

    @CELL_TO_EMPTY_MARK
    def test_meta_init_then_to_empty(self):
        class BigNet(nn.Cell):
            def __init__(self):
                super().__init__()
                self.w = Parameter(
                    Tensor(np.ones((32, 32)), mstype.float32, device="meta"),
                    name="w",
                )

            def construct(self, x):
                return x

        model = BigNet()

        assert model.w.is_meta
        assert allocated_bytes(model.w) == 0

        model.to_empty(device="CPU")

        assert not model.w.is_meta
        assert_on_device(model.w, "CPU")
        assert allocated_bytes(model.w) == 4 * 32 * 32


class TestToEmptyRecurse:
    """Recursion behavior for Cell.to_empty."""

    @CELL_TO_EMPTY_MARK
    def test_recurse_true(self, nested_net):
        nested_net.to_empty(device="CPU", recurse=True)

        for param in nested_net.get_parameters():
            assert_on_device(param, "CPU")

    @CELL_TO_EMPTY_MARK
    def test_recurse_false(self, nested_net):
        original_inner_device = nested_net.middle.inner.w_inner.device

        nested_net.to_empty(device="CPU", recurse=False)

        assert_on_device(nested_net.w_out, "CPU")
        assert nested_net.middle.inner.w_inner.device == original_inner_device


class TestToEmptyErrors:
    """Validation and return behavior."""

    @CELL_TO_EMPTY_MARK
    def test_positional_device_rejected(self, simple_net):
        with pytest.raises(TypeError):
            simple_net.to_empty("CPU")

    @CELL_TO_EMPTY_MARK
    def test_returns_self(self, simple_net):
        result = simple_net.to_empty(device="CPU")

        assert result is simple_net
