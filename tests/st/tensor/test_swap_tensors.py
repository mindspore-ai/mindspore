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
"""Tests for Tensor swap_tensors helper."""
import weakref

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore.common import dtype as mstype
from tests.mark_utils import arg_mark


SWAP_TENSORS_MARK = arg_mark(plat_marks=['cpu_linux', 'platform_ascend'],
                             level_mark='level0',
                             card_mark='onecard',
                             mem_peak=0.01)


@pytest.fixture(autouse=True)
def pynative_mode():
    ms.set_context(mode=ms.PYNATIVE_MODE)
    yield


class TestSwapTensorsBasic:
    """Basic swap_tensors behavior."""

    @SWAP_TENSORS_MARK
    def test_basic_swap_preserves_id(self):
        """Test swap_tensors preserves Python object ids."""
        from mindspore.common.tensor import swap_tensors

        a = Tensor(np.ones((2, 2)), mstype.float32)
        b = Tensor(np.zeros((2, 2)), mstype.float32)
        id_a, id_b = id(a), id(b)

        swap_tensors(a, b)

        assert id(a) == id_a
        assert id(b) == id_b
        np.testing.assert_array_equal(a.asnumpy(), np.zeros((2, 2)))
        np.testing.assert_array_equal(b.asnumpy(), np.ones((2, 2)))

    @SWAP_TENSORS_MARK
    def test_swap_preserves_external_refs(self):
        """Test external references observe swapped tensor contents."""
        from mindspore.common.tensor import swap_tensors

        a = Tensor(np.ones((2, 2)), mstype.float32)
        b = Tensor(np.zeros((2, 2)), mstype.float32)
        a_ref = a

        swap_tensors(a, b)

        np.testing.assert_array_equal(a_ref.asnumpy(), np.zeros((2, 2)))


class TestSwapTensorsValidation:
    """Validation paths for swap_tensors."""

    @SWAP_TENSORS_MARK
    def test_weakref_t1_errors(self):
        """Test weak references on t1 are rejected."""
        from mindspore.common.tensor import swap_tensors

        a = Tensor(np.ones((2, 2)), mstype.float32)
        b = Tensor(np.zeros((2, 2)), mstype.float32)
        _ = weakref.ref(a)

        with pytest.raises(RuntimeError, match="weakref"):
            swap_tensors(a, b)

    @SWAP_TENSORS_MARK
    def test_weakref_t2_errors(self):
        """Test weak references on t2 are rejected."""
        from mindspore.common.tensor import swap_tensors

        a = Tensor(np.ones((2, 2)), mstype.float32)
        b = Tensor(np.zeros((2, 2)), mstype.float32)
        _ = weakref.ref(b)

        with pytest.raises(RuntimeError, match="weakref"):
            swap_tensors(a, b)


class TestSwapTensorsSlotHandling:
    """Python state handling for swap_tensors."""

    @SWAP_TENSORS_MARK
    def test_swap_updates_dict_state(self):
        """Test Python __dict__ state is swapped."""
        from mindspore.common.tensor import swap_tensors

        class WrapA(Tensor):
            pass

        a = WrapA(Tensor(np.ones((2, 2)), mstype.float32))
        b = WrapA(Tensor(np.zeros((2, 2)), mstype.float32))
        a.meta = "A"
        b.meta = "B"

        swap_tensors(a, b)

        assert a.meta == "B"
        assert b.meta == "A"


class TestSwapTensorsSubsequentOps:
    """Behavior after swap_tensors returns."""

    @SWAP_TENSORS_MARK
    def test_swap_then_add(self):
        """Test tensor operations work after swapping contents."""
        from mindspore.common.tensor import swap_tensors

        a = Tensor(np.ones((2, 2)), mstype.float32)
        b = Tensor(np.full((2, 2), 5.0), mstype.float32)

        swap_tensors(a, b)

        result = a + 1
        np.testing.assert_array_equal(result.asnumpy(), np.full((2, 2), 6.0))
