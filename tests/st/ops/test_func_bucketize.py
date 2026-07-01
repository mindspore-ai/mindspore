# Copyright 2023 Huawei Technologies Co., Ltd
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
"""Test for ops.bucketize."""
from tests.mark_utils import arg_mark
import numpy as np
import pytest
import torch

import mindspore as ms
from mindspore import Tensor, nn, ops


class Net(nn.Cell):
    def construct(self, input_x, boundaries, right):
        return ops.bucketize(input_x, boundaries, right=right)


def torch_boundaries(boundaries, ms_dtype):
    """Build a torch boundary tensor whose dtype matches the dtype ops.bucketize uses internally:
    float64 inputs keep float64, others are promoted to float32 (aligning with torch cpu)."""
    torch_dtype = torch.float64 if ms_dtype == ms.float64 else torch.float32
    return torch.tensor(boundaries, dtype=torch_dtype)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level2',
          card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_net(mode):
    """
    Feature: test ops.bucketize
    Description: verify the result of bucketize
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()

    values = Tensor([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]])
    boundaries = [1, 2, 3, 4, 5, 6]

    expected_result = np.array([[[0, 2, 4], [1, 3, 5]], [[0, 1, 2], [3, 4, 5]]], np.int32)
    output = net(values, boundaries, right=False)
    assert np.allclose(output.asnumpy(), expected_result)

    expected_result = np.array([[[1, 3, 5], [2, 4, 6]], [[1, 2, 3], [4, 5, 6]]], np.int32)
    output = net(values, boundaries, right=True)
    assert np.allclose(output.asnumpy(), expected_result)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bucketize_float_boundary_window(mode):
    """
    Feature: test ops.bucketize
    Description: verify values lying just above a boundary are classified correctly when right=False.
        Covers the regression where a value in (boundary, boundary + 1e-6) was misclassified.
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()
    # 0.40000056883997415 falls just above the 0.4 boundary; right=False => count(boundary < x) = 2.
    values_np = np.array([0.4, 0.40000056883997415, 0.5], np.float64)
    values = Tensor(values_np, ms.float64)
    boundaries = [0.4, 0.4]
    expected = torch.bucketize(torch.from_numpy(values_np), torch_boundaries(boundaries, ms.float64),
                               right=False).numpy()
    output = net(values, boundaries, right=False)
    assert output.dtype == ms.int32
    assert np.array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bucketize_right_true_exact_boundary(mode):
    """
    Feature: test ops.bucketize
    Description: verify the right=True semantics (count(boundary <= x)) for a value exactly equal to a
        boundary, which must be counted.
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()
    # 0.4 equals the boundary; right=True counts boundaries <= x, so 0.4 maps to 2.
    values_np = np.array([0.4, 0.5, 1.0], np.float64)
    values = Tensor(values_np, ms.float64)
    boundaries = [0.4, 0.4]
    expected = torch.bucketize(torch.from_numpy(values_np), torch_boundaries(boundaries, ms.float64),
                               right=True).numpy()
    output = net(values, boundaries, right=True)
    assert output.dtype == ms.int32
    assert np.array_equal(output.asnumpy(), expected)


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bucketize_supported_dtypes(mode):
    """
    Feature: test ops.bucketize
    Description: verify bucketize works for all supported input dtypes (int32/int64/float32/float64/
        int16/float16) and always returns int32, matching torch cpu.
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()
    boundaries = [2.0, 2.0]
    dtype_cases = [
        (np.float64, ms.float64, torch.float64),
        (np.float32, ms.float32, torch.float32),
        (np.float16, ms.float16, torch.float16),
        (np.int64, ms.int64, torch.int64),
        (np.int32, ms.int32, torch.int32),
        (np.int16, ms.int16, torch.int16),
    ]
    for np_dtype, ms_dtype, torch_dtype in dtype_cases:
        if np.issubdtype(np_dtype, np.floating):
            values_np = np.array([-1.0, 0.0, 1.9, 2.0, 2.1, 5.0], np_dtype)
        else:
            values_np = np.array([-2, 0, 1, 2, 5, 10], np_dtype)
        values = Tensor(values_np, ms_dtype)
        for right in [False, True]:
            expected = torch.bucketize(torch.from_numpy(values_np).to(torch_dtype),
                                       torch_boundaries(boundaries, ms_dtype), right=right).numpy()
            output = net(values, boundaries, right=right)
            assert output.dtype == ms.int32, f"dtype {ms_dtype} right={right} output dtype {output.dtype}"
            assert np.array_equal(output.asnumpy(), expected), f"dtype {ms_dtype} right={right} mismatch"


@arg_mark(plat_marks=['platform_ascend910b', 'cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', mem_peak=0.01)
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_bucketize_int64_large_value(mode):
    """
    Feature: test ops.bucketize
    Description: verify int64 inputs with large values (beyond float32 exact range) match torch cpu,
        which promotes int to float32 for comparison against float boundaries.
    Expectation: assertion success
    """
    ms.set_context(mode=mode)
    net = Net()
    # 2**40 exceeds the float32 exact range (2**24), so int-to-float32 promotion loses precision.
    big = 2 ** 40
    values_np = np.array([big - 2, big - 1, big, big + 1, big + 2], np.int64)
    values = Tensor(values_np, ms.int64)
    boundaries = [float(big - 1), float(big + 1)]
    for right in [False, True]:
        expected = torch.bucketize(torch.from_numpy(values_np), torch_boundaries(boundaries, ms.int64),
                                   right=right).numpy()
        output = net(values, boundaries, right=right)
        assert output.dtype == ms.int32
        assert np.array_equal(output.asnumpy(), expected)
