"""
# Copyright 2020-2025 Huawei Technologies Co., Ltd
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
import numpy as np

import mindspore as ms
from mindspore import context, Tensor, ops
from mindspore import nn
from mindspore.ops import GradOperation

from tests.mark_utils import arg_mark
from tests.st.utils import test_utils


context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


@test_utils.run_with_cell
def median_with_resize(x1: Tensor, x2: Tensor, global_median=False, axis=0, keep_dims=False, ignore_nan=False):
    op = ops.Median(global_median=global_median, axis=axis, keep_dims=keep_dims, ignore_nan=ignore_nan)
    return op(x1), op(x2)

def median_global_fn(x: Tensor, ignore_nan: bool = False) -> Tensor:
    """Directly wrap global Median op (no Cell needed), return median value (ignore indices)"""
    op = ops.Median(global_median=True, ignore_nan=ignore_nan)
    out, _ = op(x)  # Only return median value for gradient computation
    return out

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0', card_mark='onecard',
          essential_mark='essential')
def test_median_op():
    """
    Feature: Median CPU ops.
    Description: Tests Median CPU op with float32 only.
    Expectation: expect correct result.
    """
    _test_median_global(np.float32)
    _test_median_axis_without_nan(np.float32)
    _test_median_axis_with_nan(np.float32)
    #_test_median_global_grad(np.float32)

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
          essential_mark='unessential')
def test_median_op_types():
    """
    Feature: Median CPU ops.
    Description: Tests Median CPU op with all supported types.
    Expectation: expect correct result.
    """
    for np_type in [np.int16, np.int32, np.int64, np.float64]:
        _test_median_global(np_type)
        _test_median_axis_without_nan(np_type)
        if np.issubdtype(np_type, np.floating):
            _test_median_axis_with_nan(np_type)


def _test_median_global(np_type) -> None:
    # test without any NaN first
    x1_np = _random((1, 10), [9], np_type, 5, (0,))
    x2_np = _random((-7, 8), (3, 5), np_type, 0, (-1, -1))
    _assert_median_global(np_type, False, x1_np, x2_np, 5, 0)

    # test with empty input, currently it returns 0 fot int series and NaN for float series
    x = Tensor(np.array([], dtype=np_type).reshape((2, 3, 5, 0, 4)))
    for ignore in (True, False):
        padding = Tensor(x1_np.copy().astype(np_type))
        exp = np.nan if np.issubdtype(np_type, np.floating) else 0
        out = median_with_resize(padding, x, global_median=True, ignore_nan=ignore)[1][0].numpy()
        assert np.array_equal(out, exp, equal_nan=True)

    # test with NaN
    if not np.issubdtype(np_type, np.floating):
        return
    x1_np = _random((1, 10), [9], np_type, 4, (-2,))
    x1_np[np.isin(x1_np, (5, 6))] = np.nan
    x2_np = _random((-7, 8), (3, 5), np_type, 0, (0, -1))
    x2_np[np.isin(x2_np, range(4, 8))] = np.nan
    _assert_median_global(np_type, False, x1_np, x2_np, np.nan, np.nan)  # do not ignore -> get NaN
    _assert_median_global(np_type, True, x1_np, x2_np, 4, -2)  # ignore


def _test_median_axis_with_nan(np_type) -> None:
    """This sub test only tests float types (float16/32/64) because int value do not have NaN."""
    x1_np = _random((0, 9), [1, 1, 9], np_type, 3, (0, 0, -1))
    x1_np[(x1_np == 5) | (x1_np == 6)] = np.nan
    x2_np = _random_axis(2, (4, 3, 6), np_type, (1, 7))
    x2_np[x2_np == 2] = np.nan
    axis = -1

    # run without ignore_nan: got all NaN and do not judge its index
    x1, x2 = [Tensor(x.copy().astype(np_type)) for x in (x1_np, x2_np)]
    (values1, indices1), (values2, indices2) = median_with_resize(x1, x2, axis=axis, ignore_nan=False, keep_dims=True)
    _assert_values(x1_np, x1, values1, np.array([[[np.nan]]], dtype=np_type))
    _assert_values(x2_np, x2, values2, np.full([4, 3, 1], np.nan, np_type))
    _assert_indices(indices1, x1_np, axis, True, np.nan)
    _assert_indices(indices2, x2_np, axis, True, np.nan)

    # run with ignore_nan: return the median after ignoring NaN
    x1, x2 = [Tensor(x.copy().astype(np_type)) for x in (x1_np, x2_np)]
    (values1, indices1), (values2, indices2) = median_with_resize(x1, x2, axis=axis, ignore_nan=True, keep_dims=True)
    _assert_values(x1_np, x1, values1, np.array([[[3]]], dtype=np_type))
    _assert_values(x2_np, x2, values2, np.full([4, 3, 1], 4, np_type))
    _assert_indices(indices1, x1_np, axis, True, 3)
    _assert_indices(indices2, x2_np, axis, True, 4)


def _test_median_axis_without_nan(np_type) -> None:
    x1_np = _random((11, 28, 2), [1, 9], np_type, 19, (0, -1))
    x2_np = _random_axis(1, (3, 5, 4), np_type, (1, 6))
    x1, x2 = [Tensor(x.copy().astype(np_type)) for x in (x1_np, x2_np)]
    (values1, indices1), (values2, indices2) = median_with_resize(x1, x2, axis=1, ignore_nan=False, keep_dims=False)
    _assert_values(x1_np, x1, values1, np.array([19], dtype=np_type))
    _assert_values(x2_np, x2, values2, np.full([3, 4], 3, np_type))
    _assert_indices(indices1, x1_np, 1, False, 19)
    _assert_indices(indices2, x2_np, 1, False, 3)

    # test empty tensor
    x2_np = np.array([[[[]]]], dtype=np_type)
    x1, x2 = [Tensor(x.copy()) for x in (x1_np, x2_np)]
    values, indices = median_with_resize(x1, x2, axis=1, ignore_nan=False, keep_dims=False)[1]
    assert list(values.shape) == [1, 1, 0]
    assert list(indices.shape) == [1, 1, 0]
    values, indices = median_with_resize(x1, x2, axis=1, ignore_nan=True, keep_dims=False)[1]
    assert list(values.shape) == [1, 1, 0]
    assert list(indices.shape) == [1, 1, 0]

def _assert_median_global(np_type, ignore_nan: bool, x1_np: np.ndarray, x2_np: np.ndarray, exp_1, exp_2) -> None:
    x1, x2 = [Tensor(x.copy().astype(np_type)) for x in (x1_np, x2_np)]
    out = median_with_resize(x1, x2, global_median=True, ignore_nan=ignore_nan)
    out1: Tensor = out[0][0]
    out2: Tensor = out[1][0]
    _assert_values(x1_np, x1, out1, exp_1)
    _assert_values(x2_np, x2, out2, exp_2)


def _assert_values(x_np: np.ndarray, x: Tensor, out: Tensor, out_exp):
    assert list(out.shape) == ([] if not isinstance(out_exp, np.ndarray) else list(out_exp.shape))
    assert out.dtype == x.dtype
    assert np.array_equal(out.numpy(), out_exp, equal_nan=True)
    assert np.array_equal(x_np, x.numpy(), equal_nan=True), "Input Tensor changed"


def _assert_indices(indices: Tensor, x_np: np.ndarray, axis: int, keep_dims: bool, mid_val) -> None:
    assert indices.dtype == ms.int64
    axis = axis + x_np.ndim if axis < 0 else axis
    x_shape = list(x_np.shape)
    exp_shape = x_shape[:axis] + ([1] if keep_dims else []) + x_shape[axis + 1:]
    assert list(indices.shape) == exp_shape
    i_np: np.ndarray = indices.numpy()
    if not keep_dims:
        i_np = i_np.reshape(x_shape[:axis] + [1] + x_shape[axis + 1:])
    np.testing.assert_array_equal(np.take_along_axis(x_np, i_np, axis), mid_val)


def _rand_arange(ranges: "tuple[int, ...]", np_type) -> np.ndarray:
    x = np.arange(*ranges).astype(np_type)
    np.random.shuffle(x)
    return x


def _random(ranges: "tuple[int, ...]", shape: "list[int]", np_type,
            find_val: int, swap_to: "tuple[int, ...]") -> np.ndarray:
    """Generate a random ndarray with non-repeated element and let `find_val` at pos `swap_to`."""
    x = _rand_arange(ranges, np_type).reshape(shape)
    x[x == find_val] = x[swap_to]
    x[swap_to] = find_val
    return x


def _random_axis(axis: int, shape: "list[int]", np_type, every_axis_range: "tuple[int, ...]") -> np.ndarray:
    return np.apply_along_axis(lambda _: _rand_arange(every_axis_range, np_type), axis, np.zeros(shape, dtype=np_type))

class GlobalMedianNet(nn.Cell):
    def __init__(self, ignore_nan=False):
        super().__init__()
        self.median = ops.Median(global_median=True, ignore_nan=ignore_nan)

    def construct(self, x):
        value, _ = self.median(x)
        return value

class GradientChecker(nn.Cell):
    def __init__(self, network):
        super().__init__()
        self.grad_op = GradOperation(get_all=True, sens_param=False)
        self.network = network

    def construct(self, x):
        gradient_function = self.grad_op(self.network)
        return gradient_function(x)[0]

@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1', card_mark='onecard',
        essential_mark='unessential')
def test_median_grad_global():
    """
    Feature: Global Median gradient CPU ops.
    Description: Tests gradient computation for global median with various data types and scenarios.
    Expectation: Gradient values match expected distribution.
    """
    test_cases = [
        # (input_data, expected_grad, data_type, ignore_nan, description)
        # Case 1: No duplicate median (odd number of elements)
        (np.array([1, 2, 3, 4, 5]), np.array([0, 0, 1, 0, 0], dtype=np.float32), np.float32, False,
            "no_duplicates_float32"),
        # Case 2: Duplicate median values
        (np.array([1, 2, 2, 3, 4]), np.array([0, 0.5, 0.5, 0, 0], dtype=np.float32), np.float32, False,
            "duplicates_float32"),
        # Case 3: All elements identical
        (np.array([5, 5, 5, 5, 5]), np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float32), np.float32, False,
            "all_identical_float32"),
        # Case 4: Integer type (output gradient should be float32)
        (np.array([10, 20, 30, 40, 30]), np.array([0, 0, 0, 0, 0], dtype=np.float32), np.int32, False,
            "integer_type"),
        # Case 5: 2D tensor with duplicates
        (np.array([[1, 1, 2], [2, 2, 3], [3, 4, 4]]), np.array([[0, 0, 0.3333333], [0.3333333, 0.3333333, 0],
            [0, 0, 0]], dtype=np.float32).flatten(), np.float32, False, "2d_tensor"),
        # Case 6: With NaN values (ignored)
        (np.array([1, np.nan, 3, np.nan, 5], dtype=np.float32), np.array([0, 0, 1, 0, 0], dtype=np.float32),np.float32,
            True, "with_nan_ignored"),
    ]

    for case in test_cases:
        input_data, expected_grad, dtype, ignore_nan, desc = case
        net = GlobalMedianNet(ignore_nan=ignore_nan)
        grad_checker = GradientChecker(net)
        input_tensor = Tensor(input_data.astype(dtype))
        actual_grad = grad_checker(input_tensor).asnumpy()

        if input_data.ndim > 1:
            expected_grad = expected_grad.reshape(input_data.shape)
        if np.issubdtype(dtype, np.floating):
            np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5,
                err_msg=f"Failed for case: {desc}, dtype: {dtype}")
        else:
            np.testing.assert_allclose(actual_grad, expected_grad, atol=1e-5, rtol=1e-5,
                err_msg=f"Failed for case: {desc}, dtype: {dtype}")
