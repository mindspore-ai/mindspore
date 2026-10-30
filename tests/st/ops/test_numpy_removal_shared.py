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
"""Regression coverage for shared rules after removing the NumPy API package."""
import numpy as np
import pytest

import mindspore as ms
from mindspore import Tensor, nn, ops
from mindspore.ops._grad_experimental.grad_math_ops import get_bprop_fft_with_size
from tests.mark_utils import arg_mark


@pytest.fixture(autouse=True)
def restore_mode():
    """Do not change the mode of tests collected after this module."""
    mode = ms.get_context('mode')
    yield
    ms.set_context(mode=mode)


class MixedOutputs(nn.Cell):
    def construct(self, x, y):
        return x + y, y * 2


class MappedMixedOutputs(nn.Cell):
    def __init__(self):
        super().__init__()
        self.mapped = ops.vmap(MixedOutputs(), in_axes=(0, None), out_axes=(-1, 1))

    def construct(self, x, y):
        return self.mapped(x, y)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_vmap_output_axis_and_unmapped_output(mode):
    """
    Feature: vmap shared axis handling.
    Description: Move a batched output axis and broadcast an output with no batch axis.
    Expectation: Both outputs match a per-example evaluation, including their shapes.
    """
    ms.set_context(mode=mode)
    x = np.arange(12, dtype=np.float32).reshape(3, 4)
    y = np.arange(4, dtype=np.float32)
    first, second = MappedMixedOutputs()(Tensor(x), Tensor(y))
    np.testing.assert_array_equal(first.asnumpy(), np.moveaxis(x + y, 0, -1))
    np.testing.assert_array_equal(second.asnumpy(), np.broadcast_to((y * 2)[:, None], (4, 3)))


class Gather(nn.Cell):
    def construct(self, x, indices):
        return ops.gather(x, indices, 0)


class MappedGather(nn.Cell):
    def __init__(self):
        super().__init__()
        self.gather = ops.vmap(Gather(), in_axes=(0, 0), out_axes=0)

    def construct(self, x, indices):
        return self.gather(x, indices)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('index_type', [np.int32, np.int64])
def test_vmap_gather_both_inputs_batched(mode, index_type):
    """
    Feature: Gather vmap index offsets.
    Description: Exercise the arange and scalar-where path with both arguments batched.
    Expectation: Per-batch indices address the correct rows without dtype changes to output.
    """
    ms.set_context(mode=mode)
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    indices = np.array([[2, 0], [1, 2]], dtype=index_type)
    result = MappedGather()(Tensor(x), Tensor(indices))
    expected = np.stack([x[i, indices[i]] for i in range(2)])
    assert result.dtype == ms.float32
    np.testing.assert_array_equal(result.asnumpy(), expected)


class ReverseSequence(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reverse = ops.ReverseSequence(seq_dim=1, batch_dim=0)

    def construct(self, x, lengths):
        return self.reverse(x, lengths)


class MappedReverseSequence(nn.Cell):
    def __init__(self):
        super().__init__()
        self.reverse = ops.vmap(ReverseSequence(), in_axes=(1, 0), out_axes=1)

    def construct(self, x, lengths):
        return self.reverse(x, lengths)


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
def test_vmap_reverse_sequence_multiple_axes(mode):
    """
    Feature: Multiple-axis move in the ReverseSequence vmap rule.
    Description: Map a non-leading input axis with independently batched sequence lengths.
    Expectation: The prefix of each sequence is reversed without permuting other elements.
    """
    ms.set_context(mode=mode)
    x = np.arange(24, dtype=np.float32).reshape(3, 2, 4)
    lengths = np.array([[1, 3, 4], [2, 0, 3]], dtype=np.int32)
    expected = x.copy()
    for batch in range(2):
        for row in range(3):
            size = lengths[batch, row]
            expected[row, batch, :size] = x[row, batch, :size][::-1]
    result = MappedReverseSequence()(Tensor(x), Tensor(lengths))
    np.testing.assert_array_equal(result.asnumpy(), expected)


class RealFftPythonBprop(nn.Cell):
    """Invoke the Python registration explicitly, rather than a C++ bprop substitute."""
    def __init__(self, signal_ndim):
        super().__init__()
        self.signal_ndim = signal_ndim
        self.fft = ops.FFTWithSize(signal_ndim=signal_ndim, inverse=False, real=True,
                                   norm='backward', onesided=True)
        self.backward = get_bprop_fft_with_size(self.fft)

    def construct(self, x, dout):
        out = self.fft(x)
        return self.backward(x, self.signal_ndim, False, True, 0, True, (), out, dout)[0]


@arg_mark(plat_marks=['cpu_linux'], level_mark='level1', card_mark='onecard')
@pytest.mark.parametrize('mode', [ms.GRAPH_MODE, ms.PYNATIVE_MODE])
@pytest.mark.parametrize('shape', [(5,), (6,), (3, 4)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_real_fft_python_bprop_ranges_and_phase(mode, shape, dtype):
    """
    Feature: Shared real FFT Python backward rule.
    Description: Cover odd/even ranges and the 2D phase matrix with an explicit NumPy Fourier basis.
    Expectation: Backward values and dtype agree with the independently evaluated real linear map.
    """
    ms.set_context(mode=mode)
    data = np.arange(np.prod(shape), dtype=dtype).reshape(shape) / 10
    complex_dtype = np.complex64 if dtype == np.float32 else np.complex128
    output_shape = np.fft.rfftn(data).shape
    dout = (np.arange(np.prod(output_shape)).reshape(output_shape) + 1).astype(complex_dtype)
    expected = np.zeros(shape, dtype=dtype)
    for index in np.ndindex(shape):
        basis = np.zeros(shape, dtype=dtype)
        basis[index] = 1
        expected[index] = np.sum(np.real(np.conj(dout) * np.fft.rfftn(basis)))
    result = RealFftPythonBprop(len(shape))(Tensor(data), Tensor(dout))
    assert result.dtype == (ms.float32 if dtype == np.float32 else ms.float64)
    np.testing.assert_allclose(result.asnumpy(), expected, atol=2e-5, rtol=2e-5)
