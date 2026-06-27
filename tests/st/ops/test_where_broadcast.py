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
"""
Regression tests for issue 29437: ops.where returns wrong shape/values in
GRAPH mode when the three inputs are constant tensors that require
broadcasting. The bug lives in SelectFrontendFuncImpl::InferValue which is
invoked as constant folding during graph compilation and originally did not
handle broadcasting; the fix defers folding to the runtime Select kernel when
shapes differ.
"""
import numpy as np
import pytest

from mindspore import Tensor, context, nn, ops
from tests.mark_utils import arg_mark


class WhereConstNet(nn.Cell):
    """Constructs constant Tensors inside ``construct``.

    This is the path that triggers SelectFrontendFuncImpl::InferValue
    constant folding in GRAPH mode.
    """

    def __init__(self, cond_np, x_np, y_np):
        super().__init__()
        self._cond_np = cond_np
        self._x_np = x_np
        self._y_np = y_np

    def construct(self):
        cond = Tensor(self._cond_np)
        x = Tensor(self._x_np)
        y = Tensor(self._y_np)
        return ops.where(cond, x, y)


class WhereInputNet(nn.Cell):
    """Receives the three inputs as arguments (no constant folding)."""

    def construct(self, cond, x, y):
        return ops.where(cond, x, y)


def _expected(cond_np, x_np, y_np):
    return np.where(cond_np, x_np, y_np)


def _run_and_check(mode, cond_np, x_np, y_np):
    context.set_context(mode=mode)
    expect = _expected(cond_np, x_np, y_np)

    # Path 1: constants built inside construct (triggers InferValue folding in GRAPH mode).
    out_const = WhereConstNet(cond_np, x_np, y_np)().asnumpy()
    assert out_const.shape == expect.shape, (
        f"const-in-construct shape mismatch: got {out_const.shape}, expect {expect.shape}")
    assert np.array_equal(out_const, expect), (
        f"const-in-construct value mismatch: got {out_const.tolist()}, expect {expect.tolist()}")

    # Path 2: inputs passed as arguments (does NOT trigger folding).
    out_input = WhereInputNet()(Tensor(cond_np), Tensor(x_np), Tensor(y_np)).asnumpy()
    assert out_input.shape == expect.shape, (
        f"input-arg shape mismatch: got {out_input.shape}, expect {expect.shape}")
    assert np.array_equal(out_input, expect), (
        f"input-arg value mismatch: got {out_input.tolist()}, expect {expect.tolist()}")


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_three_way_broadcast(mode):
    """
    Feature: ops.where broadcasting
    Description: Issue 29437 reproducer. cond(2,1), x(1,3), y(2,3) broadcast to (2,3).
                 In GRAPH mode with constant inputs the InferValue folding path
                 used to return shape (1,3) with wrong values.
    Expectation: Output matches numpy.where on shape and values in both modes.
    """
    cond_np = np.array([[True], [False]], dtype=np.bool_)
    x_np = np.ones((1, 3), dtype=np.float32)
    y_np = np.zeros((2, 3), dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_cond_broadcast(mode):
    """
    Feature: ops.where broadcasting
    Description: Only condition needs broadcasting. cond(2,1), x(2,3), y(2,3).
    Expectation: Output matches numpy.where in both modes.
    """
    cond_np = np.array([[True], [False]], dtype=np.bool_)
    x_np = np.full((2, 3), 1.0, dtype=np.float32)
    y_np = np.full((2, 3), 2.0, dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_input_broadcast(mode):
    """
    Feature: ops.where broadcasting
    Description: Only input needs broadcasting. cond(2,3), x(1,3), y(2,3).
    Expectation: Output matches numpy.where in both modes.
    """
    cond_np = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    x_np = np.array([[10.0, 20.0, 30.0]], dtype=np.float32)
    y_np = np.full((2, 3), -1.0, dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_other_broadcast(mode):
    """
    Feature: ops.where broadcasting
    Description: Only other (y) needs broadcasting. cond(2,3), x(2,3), y(1,3).
    Expectation: Output matches numpy.where in both modes.
    """
    cond_np = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    x_np = np.full((2, 3), 7.0, dtype=np.float32)
    y_np = np.array([[-7.0, -8.0, -9.0]], dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_no_broadcast_constant_folding(mode):
    """
    Feature: ops.where constant folding
    Description: Three inputs with identical shape (2,3) still go through the
                 InferValue constant folding path in GRAPH mode. This guards
                 the non-broadcast fast path from regression.
    Expectation: Output matches numpy.where in both modes.
    """
    cond_np = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    x_np = np.full((2, 3), 1.0, dtype=np.float32)
    y_np = np.full((2, 3), 2.0, dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos', 'platform_gpu', 'platform_ascend'],
          level_mark='level0', card_mark='onecard', mem_peak=2048.01)
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_where_scalar_like_broadcast(mode):
    """
    Feature: ops.where broadcasting
    Description: Mix shapes including a 1-element tensor. cond(2,3), x(1,), y(2,3).
    Expectation: Output matches numpy.where in both modes.
    """
    cond_np = np.array([[True, False, True], [False, True, False]], dtype=np.bool_)
    x_np = np.array([5.0], dtype=np.float32)
    y_np = np.full((2, 3), -5.0, dtype=np.float32)
    _run_and_check(mode, cond_np, x_np, y_np)
