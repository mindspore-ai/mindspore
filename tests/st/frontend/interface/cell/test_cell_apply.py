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
"""Tests for Cell._apply three-strategy engine."""
# pylint: disable=protected-access
import numpy as np

from mindspore import Tensor, Parameter, nn
from mindspore.common import dtype as mstype
from mindspore.nn.cell import _is_traceable_tensor_subclass
from tests.mark_utils import arg_mark
from tests.st.frontend.interface.cell.conftest import MinimalTraceableTensor


CELL_APPLY_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                           level_mark='level0',
                           card_mark='onecard',
                           mem_peak=0.01)


class TestApplyStrategyB:
    """Default path: plain Parameter uses .data assignment."""

    @CELL_APPLY_MARK
    def test_params_id_preserved_after_apply(self, simple_net):
        original_id = id(simple_net.w)

        simple_net._apply(lambda t: t + 1)

        assert id(simple_net.w) == original_id

    @CELL_APPLY_MARK
    def test_external_ref_sees_updates(self, simple_net):
        param_ref = simple_net.w

        simple_net._apply(lambda t: t * 2)

        np.testing.assert_allclose(param_ref.asnumpy(), np.full((4, 4), 2.0))


class TestApplyStrategyA:
    """Traceable Tensor subclass path uses swap_tensors."""

    @CELL_APPLY_MARK
    def test_traceable_tensor_subclass_requires_tensor_wrapper(self):
        class NonTensorTraceable:
            def __tensor_flatten__(self):
                return [], None

            @classmethod
            def __tensor_unflatten__(cls, inner_tensors, metadata, outer_size, outer_stride):
                return cls()

        assert _is_traceable_tensor_subclass(
            MinimalTraceableTensor(Tensor(np.ones((2, 2)), mstype.float32), payload="p")
        )
        assert not _is_traceable_tensor_subclass(Tensor(np.ones((2, 2)), mstype.float32))
        assert not _is_traceable_tensor_subclass(NonTensorTraceable())

    @CELL_APPLY_MARK
    def test_swap_preserves_subclass_and_slot(self, simple_net):
        traceable_tensor = MinimalTraceableTensor(Tensor(np.ones((4, 4)), mstype.float32), payload="old")
        simple_net.w = Parameter(traceable_tensor, name="w")
        original_id = id(simple_net.w)

        def fn(tensor):
            return MinimalTraceableTensor(tensor + 1, payload="new")

        simple_net._apply(fn)

        assert id(simple_net.w) == original_id
        assert isinstance(simple_net.w, MinimalTraceableTensor)
        assert simple_net.w._payload_slot == "new"
        np.testing.assert_allclose(simple_net.w.asnumpy(), np.full((4, 4), 2.0))


class TestApplyRecursion:
    """Recursion behavior for Cell._apply."""

    @CELL_APPLY_MARK
    def test_recurse_true_applies_to_children(self, nested_net):
        nested_net._apply(lambda t: t + 1, recurse=True)

        np.testing.assert_allclose(nested_net.w_out.asnumpy(), np.full((4, 4), 2.0))
        np.testing.assert_allclose(nested_net.middle.w_mid.asnumpy(), np.full((3, 3), 2.0))
        np.testing.assert_allclose(nested_net.middle.inner.w_inner.asnumpy(), np.full((2, 2), 2.0))

    @CELL_APPLY_MARK
    def test_recurse_false_skips_children(self, nested_net):
        nested_net._apply(lambda t: t + 1, recurse=False)

        np.testing.assert_allclose(nested_net.w_out.asnumpy(), np.full((4, 4), 2.0))
        np.testing.assert_allclose(nested_net.middle.w_mid.asnumpy(), np.full((3, 3), 1.0))
        np.testing.assert_allclose(nested_net.middle.inner.w_inner.asnumpy(), np.full((2, 2), 1.0))


class TestApplyBuffer:
    """Buffer update path."""

    @CELL_APPLY_MARK
    def test_buffer_updated(self, simple_net):
        simple_net._apply(lambda t: t + 10)

        np.testing.assert_array_equal(simple_net.buf.asnumpy(), np.array([11, 12, 13]))


class TestApplyEmptyCell:
    """Empty and optional parameter handling."""

    @CELL_APPLY_MARK
    def test_empty_cell_noop(self):
        class EmptyNet(nn.Cell):
            def construct(self, x):
                return x

        net = EmptyNet()

        result = net._apply(lambda t: t + 1)

        assert result is net

    @CELL_APPLY_MARK
    def test_none_param_skipped(self, simple_net):
        simple_net._params["optional"] = None

        simple_net._apply(lambda t: t + 1)

        assert simple_net._params["optional"] is None
