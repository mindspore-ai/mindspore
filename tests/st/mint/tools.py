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
""" test tools"""
import numpy as np

def allclose_nparray(data_expected, data_ms, rtol=0, atol=0, equal_nan=True):
    """
    Compare two numpy arrays for numerical equality with detailed error reporting.

    This function performs a comprehensive comparison between expected and actual
    numpy arrays, checking data types, shapes, and numerical values. It provides
    detailed error information when comparisons fail, including data ranges,
    maximum absolute and relative errors.

    Args:
        data_expected (numpy.ndarray): The expected/ground truth array
        data_ms (numpy.ndarray): The actual array to compare against
        rtol (float, optional): Relative tolerance for numerical comparison.
            Defaults to 0.
        atol (float, optional): Absolute tolerance for numerical comparison.
            Defaults to 0.
        equal_nan (bool, optional): Whether to treat NaN values as equal.
            Defaults to True.

    Raises:
        AssertionError: If data types don't match, shapes don't match, or
            numerical values differ beyond the specified tolerances.

    Example:
        >>> import numpy as np
        >>> expected = np.array([1.0, 2.0, 3.0])
        >>> actual = np.array([1.0, 2.001, 3.0])
        >>> allclose_nparray(expected, actual, rtol=1e-3, atol=1e-3)
        # Will pass if values are within tolerance
    """
    # First validate metadata equality (dtype and shape)
    compare_shape_dtype(data_expected, data_ms)

    try:
        np.testing.assert_allclose(data_expected, data_ms, rtol=rtol, atol=atol,
                                   equal_nan=equal_nan)
    except AssertionError as e:
        print(f"Numerical comparison failed: {e}")
        print(f"Expected data range: [{data_expected.min()}, "
              f"{data_expected.max()}]")
        print(f"Actual data range: [{data_ms.min()}, {data_ms.max()}]")
        max_abs_error = np.max(np.abs(data_expected - data_ms))
        print(f"Maximum absolute error: {max_abs_error}")

        # Calculate maximum relative error
        # Avoid division by zero by adding small epsilon
        epsilon = 1e-15
        relative_errors = np.abs(data_expected - data_ms) / (np.abs(data_expected) + epsilon)
        max_rel_error = np.max(relative_errors)
        print(f"Maximum relative error: {max_rel_error}")

        print(f"Tolerance parameters: rtol={rtol}, atol={atol}")
        raise


def compare_shape_dtype(data_expected, data_actual):
    """
    Compare only dtype and shape of two numpy arrays with detailed error reporting.

    This helper is intended for operators whose contents are unspecified or uninitialized
    (e.g., empty/empty_like). It validates metadata equality without checking values.

    Args:
        data_expected (numpy.ndarray): The expected array.
        data_actual (numpy.ndarray): The actual array to compare against.

    Raises:
        AssertionError: If data types or shapes do not match.
    """
    try:
        assert data_expected.dtype == data_actual.dtype, \
            f"Data type mismatch: expected {data_expected.dtype}, got {data_actual.dtype}"
    except AssertionError as e:
        print(f"Data type assertion failed: {e}")
        print(f"Expected dtype: {data_expected.dtype}")
        print(f"Actual dtype: {data_actual.dtype}")
        raise

    try:
        assert data_expected.shape == data_actual.shape, \
            f"Shape mismatch: expected {data_expected.shape}, got {data_actual.shape}"
    except AssertionError as e:
        print(f"Shape assertion failed: {e}")
        print(f"Expected shape: {data_expected.shape}")
        print(f"Actual shape: {data_actual.shape}")
        raise


def needs_backward(dtype):
    """Determine if data type needs backward propagation"""
    return dtype in [np.float32, np.float64, np.complex64, np.complex128,
                     np.float16]
