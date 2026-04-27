# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""CPU matmul operator tests."""

import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, context, nn, ops
from mindspore.common import dtype as mstype
from mindspore.graph.api import _pynative_executor
from mindspore.ops import operations as P

from tests.mark_utils import arg_mark


np.random.seed(100)


class MatMulNet(nn.Cell):
    """MatMul test network."""

    def __init__(self, transpose_a=False, transpose_b=False):
        super().__init__()
        self.matmul = P.MatMul(transpose_a, transpose_b)

    def construct(self, x, y):  # pylint: disable=arguments-differ
        return self.matmul(x, y)


def judge_result_correct(result, expect):
    """Assert that tensor results match the expected ndarray."""

    assert result.dtype == expect.dtype
    assert result.shape == expect.shape
    assert np.allclose(result, expect)


ARG_MARK = arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level1',
    card_mark='onecard',
)

ARG_MARK_LEVEL0 = arg_mark(
    plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'],
    level_mark='level0',
    card_mark='onecard',
)


@ARG_MARK
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_matmul_no_transpose_vec(dtype):
    """
    Feature: matrix & vec
    Description: test cases for matmul between matrix and vector
    Expectation: the result match to scipy
    """
    a = np.arange(1 * 3).reshape((1, 3)).astype(dtype)
    b = np.arange(3 * 5).reshape((3, 5)).astype(dtype)

    context.set_context(mode=context.GRAPH_MODE, device_target='CPU')
    net = MatMulNet()
    output = net(Tensor(a), Tensor(b)).asnumpy()
    expect = np.array([[25., 28., 31., 34., 37.]], dtype)
    judge_result_correct(output, expect)


def np_matmul(a: np.ndarray, b: np.ndarray, trans_a: bool, trans_b: bool):
    """Apply matmul after optional transpose handling."""

    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    return np.matmul(a, b)


@ARG_MARK
@pytest.mark.parametrize('trans_a', [True, False])
@pytest.mark.parametrize('trans_b', [True, False])
@pytest.mark.parametrize('dtype', [np.float16, np.float32, np.float64])
def test_matmul_matrix(trans_a, trans_b, dtype):
    """
    Feature: ALL To ALL
    Description: test cases for matmul for all float types and transpose args combinations
    Expectation: the result match to scipy
    """
    m, k, n = 5, 3, 4
    a = np.random.random((m, k)).astype(dtype)
    b = np.random.random((k, n)).astype(dtype)
    if trans_a:
        a = a.T
    if trans_b:
        b = b.T
    expect = np_matmul(a, b, trans_a, trans_b)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    net = MatMulNet(transpose_a=trans_a, transpose_b=trans_b)
    output = net(Tensor(a), Tensor(b)).asnumpy()

    judge_result_correct(output, expect)


@ARG_MARK
@pytest.mark.parametrize('mode', [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_matmul_tensor_api_modes(mode):
    """
    Feature: Test matmul tensor api.
    Description: Test matmul tensor api for Graph and PyNative modes.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=mode, device_target="CPU")
    x = Tensor(np.arange(2 * 3 * 4).reshape(2, 3, 4), mstype.float32)
    y = Tensor(np.arange(4 * 5).reshape(4, 5), mstype.float32)
    output = x.matmul(y)
    expected = np.array([[[70., 76., 82., 88., 94.],
                          [190., 212., 234., 256., 278.],
                          [310., 348., 386., 424., 462.]],
                         [[430., 484., 538., 592., 646.],
                          [550., 620., 690., 760., 830.],
                          [670., 756., 842., 928., 1014.]]], np.float32)
    np.testing.assert_array_equal(output.asnumpy(), expected)


@ms.jit
def jit_matmul(x, y):
    """Run matmul through JIT to cover Graph mode fallback behavior."""

    return ops.matmul(x, y)


@ARG_MARK_LEVEL0
@pytest.mark.parametrize(
    'x_np, y_np, expected',
    [
        (
            np.zeros((0, 3), dtype=np.float32),
            np.ones((3, 2), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        ),
        (
            np.empty((2, 0), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.zeros((2, 3), dtype=np.float32),
        ),
        (
            np.ones((2, 3), dtype=np.float32),
            np.empty((3, 0), dtype=np.float32),
            np.empty((2, 0), dtype=np.float32),
        ),
    ],
)
def test_ops_matmul_zero_dimension_consistent_between_modes(x_np, y_np, expected):
    """
    Feature: ops.matmul on CPU.
    Description: Zero-dimension inputs should keep the same shape/value semantics
    in Graph and PyNative modes.
    Expectation: Graph and PyNative outputs match numpy and each other.
    """
    x = Tensor(x_np)
    y = Tensor(y_np)

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    pynative_out = ops.matmul(x, y)

    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    graph_out = jit_matmul(x, y)

    np.testing.assert_array_equal(pynative_out.asnumpy(), expected)
    np.testing.assert_array_equal(graph_out.asnumpy(), expected)
    assert pynative_out.shape == expected.shape
    assert graph_out.shape == expected.shape


@ARG_MARK
def test_matmul_dtypes():
    """
    Feature: Test matmul dtypes.
    Description: Test matmul dtypes for Graph mode.
    Expectation: The result match to the expect value.
    """
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")
    m = 3
    n = 3
    k = 4
    x_np = np.random.randn(m * k).astype(np.float32)
    y_np = np.random.randn(k * n).astype(np.float32)
    x_np.shape = m, k
    y_np.shape = k, n
    matmul = P.MatMul()
    valid_dtypes = (
        mstype.uint8, mstype.int8, mstype.int16, mstype.int32, mstype.int64,
        mstype.float16, mstype.float32, mstype.float64, mstype.complex64,
        mstype.complex128,
    )
    invalid_dtypes = (mstype.qint4x2, mstype.float8_e4m3fn, mstype.float8_e5m2, mstype.hifloat8)
    all_dtypes = (dtype for dtype in mstype.all_types if dtype not in invalid_dtypes)
    for dtype in all_dtypes:
        # bfloat16 is not supported yet
        if dtype == mstype.bfloat16:
            continue
        x_ms = Tensor(x_np).astype(dtype)
        y_ms = Tensor(y_np).astype(dtype)
        if dtype in valid_dtypes:
            out = matmul(x_ms, y_ms)
            assert out.dtype == x_ms.dtype
        else:
            with pytest.raises(TypeError):
                matmul(x_ms, y_ms)
                _pynative_executor.sync()
