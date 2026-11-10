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
Compatibility tests for building with NumPy 2.x and running with either
NumPy 1.x (>= 1.25) or NumPy 2.x.

Covers:
  * Runtime version gate (``np_dtype_valid``) matches the documented ABI matrix
    in ``mindspore/ccsrc/utils/np_dtypes.cc::NumpyVersionValid``.
  * ``mindspore.bfloat16`` registration succeeds when the gate is True.
  * Round-trip ``Tensor <-> numpy.ndarray`` works across supported NumPy
    versions, including the bfloat16 custom dtype.
  * Production code paths that previously used NumPy 1.x-only aliases
    (``np.Inf``, ``np.float_``, ``np.trapz``) keep importing and working.
"""
from __future__ import annotations

import importlib

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor
from mindspore._c_expression.np_dtypes import np_dtype_valid

from tests.mark_utils import arg_mark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _numpy_major_minor() -> tuple[int, int]:
    parts = np.__version__.split(".")
    return int(parts[0]), int(parts[1])


def _expected_gate_result() -> bool:
    """Mirror ``NumpyVersionValid`` for the compile-time ``NPY_TARGET_VERSION=0x11``
    scenario: wheels built against NumPy 2.x support NumPy 1.25+ or 2.x at runtime;
    wheels built against NumPy 1.x support whatever range their ABI allows.

    Since this test does not know the compile-time ``NPY_API_VERSION``, the
    assertion is asymmetric: when the running NumPy is in the union of all
    supported ranges, the gate *must* be True; otherwise the test is skipped
    rather than asserting False (other compile targets may still reject it).
    """
    major, minor = _numpy_major_minor()
    if major == 2:
        return True
    if major == 1 and minor >= 25:
        return True
    return False


# ---------------------------------------------------------------------------
# Version gate
# ---------------------------------------------------------------------------


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos',
                      'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.1)
def test_np_dtype_valid_matches_running_numpy() -> None:
    """
    Feature: NumPy ABI version gate
    Description: ``np_dtype_valid`` must return True whenever the running NumPy
                 is in the documented compat matrix for a 2.x-targeted build.
    Expectation: True for NumPy 2.x and NumPy 1.25+; no crash either way.
    """
    expected = _expected_gate_result()
    if not expected:
        pytest.skip(
            f"Running NumPy {np.__version__} is outside the 1.25+/2.x union; "
            "gate result depends on compile-time NPY_API_VERSION.")
    assert np_dtype_valid(False) is True
    assert np_dtype_valid(True) is True


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos',
                      'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.1)
def test_np_dtype_valid_is_idempotent() -> None:
    """
    Feature: NumPy ABI version gate
    Description: repeated calls must return the same cached result.
    Expectation: consistent across invocations.
    """
    first = np_dtype_valid(False)
    for _ in range(10):
        assert np_dtype_valid(False) is first


# ---------------------------------------------------------------------------
# bfloat16 availability & round-trip
# ---------------------------------------------------------------------------


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos',
                      'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard', mem_peak=0.1)
def test_bfloat16_dtype_is_registered_when_valid() -> None:
    """
    Feature: bfloat16 custom NumPy dtype
    Description: when the version gate is True, ``mindspore.bfloat16`` and the
                 underlying NumPy scalar type must be importable.
    Expectation: import succeeds and ``np.dtype(bfloat16)`` returns a dtype.
    """
    if not np_dtype_valid(False):
        pytest.skip(
            f"np_dtype_valid is False for NumPy {np.__version__}; bfloat16 "
            "is intentionally not registered in this compat window.")

    from mindspore._c_expression.np_dtypes import bfloat16 as np_bfloat16
    dtype = np.dtype(np_bfloat16)
    assert dtype is not None
    # bfloat16 occupies 2 bytes regardless of NumPy major version.
    assert dtype.itemsize == 2


@arg_mark(plat_marks=['platform_ascend910b'],
          level_mark='level1',
          card_mark='onecard', mem_peak=0.5)
@pytest.mark.parametrize('mode', [ms.PYNATIVE_MODE, ms.GRAPH_MODE])
def test_bfloat16_tensor_roundtrip(mode: int) -> None:
    """
    Feature: bfloat16 Tensor <-> NumPy round-trip
    Description: Tensor(x, bf16).asnumpy() must return an ndarray whose dtype
                 round-trips through NumPy's dtype system.
    Expectation: shape/itemsize preserved across NumPy 1.x and 2.x.
    """
    if not np_dtype_valid(False):
        pytest.skip("bfloat16 not registered for current NumPy.")

    ms.set_context(mode=mode)
    data = np.ones((2, 3), dtype=np.float32)
    tensor = Tensor(data, dtype=ms.bfloat16)
    arr = tensor.asnumpy()
    assert arr.shape == (2, 3)
    assert arr.dtype.itemsize == 2
    # Cast back to float32 for value comparison (bfloat16 has limited precision).
    assert np.allclose(arr.astype(np.float32), data, atol=1e-2)


# ---------------------------------------------------------------------------
# Generic dtype round-trip (no bfloat16 dependency)
# ---------------------------------------------------------------------------


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos',
                      'platform_gpu', 'platform_ascend', 'platform_ascend910b'],
          level_mark='level0',
          card_mark='onecard', mem_peak=0.5)
@pytest.mark.parametrize(
    'np_dtype, ms_dtype',
    [
        (np.float16, ms.float16),
        (np.float32, ms.float32),
        (np.float64, ms.float64),
        (np.int8, ms.int8),
        (np.int16, ms.int16),
        (np.int32, ms.int32),
        (np.int64, ms.int64),
        (np.uint8, ms.uint8),
        (np.bool_, ms.bool_),
    ],
)
def test_standard_dtype_roundtrip(np_dtype, ms_dtype) -> None:
    """
    Feature: Standard dtype round-trip across NumPy 1.x/2.x
    Description: Build with NumPy 2.x, run with 1.25+ or 2.x: scalar types
                 must still round-trip through Tensor.asnumpy without loss.
    Expectation: dtype preserved, values equal.
    """
    data = np.array([[1, 2], [3, 4]], dtype=np_dtype)
    tensor = Tensor(data, dtype=ms_dtype)
    out = tensor.asnumpy()
    assert out.dtype == np_dtype
    assert np.array_equal(out, data)


# ---------------------------------------------------------------------------
# Regression: production modules that used removed NumPy 1.x aliases
# ---------------------------------------------------------------------------


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard')
@pytest.mark.parametrize(
    'module_name',
    [
        'mindspore.train.callback._early_stop',
        'mindspore.train.callback._reduce_lr_on_plateau',
        'mindspore.train.callback._landscape',
        'mindspore.train.metrics.auc',
        'mindspore.train.summary._summary_adapter',
        'mindspore.nn.layer.math',
        'mindspore.numpy.dtypes',
    ],
)
def test_modules_import_without_removed_aliases(module_name: str) -> None:
    """
    Feature: NumPy 2.x alias-removal regression
    Description: modules that previously used ``np.Inf`` / ``np.float_`` /
                 ``np.trapz`` must import cleanly on any supported NumPy.
    Expectation: import succeeds on NumPy 1.x and 2.x.
    """
    module = importlib.import_module(module_name)
    assert module is not None


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
          level_mark='level0',
          card_mark='onecard')
def test_auc_trapezoid_fallback_works() -> None:
    """
    Feature: auc metric NumPy 2.x compat
    Description: ``auc`` switched from ``np.trapz`` (deprecated in 2.0) to a
                 ``getattr(np, 'trapezoid', None) or np.trapz`` fallback.
    Expectation: numeric result is correct on both NumPy 1.x and 2.x.
    """
    from mindspore.train.metrics.auc import auc

    x = np.array([0.0, 0.5, 1.0])
    y = np.array([0.0, 1.0, 1.0])
    # Monotonically increasing x, triangle + rectangle = 0.25 + 0.5 = 0.75.
    result = auc(x, y, reorder=False)
    assert np.isclose(result, 0.75, atol=1e-6)
