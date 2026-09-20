# Copyright 2022 Huawei Technologies Co., Ltd
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
"""STFT operator tests for CPU."""

import numpy as np
import pytest

from mindspore import Tensor, context, nn, ops
from mindspore.common.api import _pynative_executor
from mindspore.ops import functional as F
from tests.mark_utils import arg_mark


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.00001, 0.00001, equal_nan=True)

def _numpy_stft(x_np, n_fft, hop_length, win_length, window_np=None,
                normalized=False, onesided=True, return_complex=False):
    """Generate the NumPy STFT reference output for center=False."""
    window_np = np.ones(win_length, dtype=x_np.dtype) if window_np is None else window_np
    common_dtype = np.result_type(x_np.dtype, window_np.dtype)
    x_np = x_np.astype(common_dtype)
    window_np = window_np.astype(common_dtype)
    if win_length < n_fft:
        left = (n_fft - win_length) // 2
        window_np = np.pad(window_np, (left, n_fft - win_length - left))

    frame_count = 1 + (x_np.shape[-1] - n_fft) // hop_length
    frames = np.stack([x_np[..., i * hop_length:i * hop_length + n_fft]
                       for i in range(frame_count)], axis=-2)
    spectrum = np.fft.fft(frames * window_np, axis=-1, norm="ortho" if normalized else None)
    if onesided:
        spectrum = spectrum[..., :n_fft // 2 + 1]
    spectrum = np.swapaxes(spectrum, -1, -2)
    complex_dtype = np.complex128 if common_dtype in (np.float64, np.complex128) else np.complex64
    spectrum = spectrum.astype(complex_dtype)
    if return_complex:
        return spectrum
    return np.stack((spectrum.real, spectrum.imag), axis=-1)


def _compare_stft_with_numpy(x_np, n_fft, hop_length, win_length, window_np=None,
                             normalized=False, onesided=True, return_complex=False):
    """Compare MindSpore CPU STFT output with the NumPy reference."""
    window = Tensor(window_np) if window_np is not None else None
    output = F.stft(Tensor(x_np), n_fft, hop_length=hop_length, win_length=win_length, window=window,
                    center=False, normalized=normalized, onesided=onesided, return_complex=return_complex)
    expect = _numpy_stft(x_np, n_fft, hop_length, win_length, window_np,
                         normalized, onesided, return_complex)
    output_np = output.asnumpy()
    assert output_np.shape == expect.shape
    assert output_np.dtype == expect.dtype
    np.testing.assert_allclose(output_np, expect, rtol=1e-4, atol=1e-4)



@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_stft_small_input(mode):
    """
    Feature: STFT CPU kernel.
    Description: Run a small real input through the unified STFT entry in graph and pynative modes.
    Expectation: The result matches numpy.fft.rfft.
    """
    context.set_context(mode=mode, device_target="CPU")
    x_np = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    window_np = np.ones(4, dtype=np.float32)
    output = F.stft(Tensor(x_np), 4, hop_length=4, win_length=4, window=Tensor(window_np), center=False,
                    onesided=True, return_complex=False)

    expected_complex = np.fft.rfft(x_np[:4] * window_np)
    expected = np.stack((expected_complex.real, expected_complex.imag), axis=-1).astype(np.float32)[:, None, :]
    np.testing.assert_allclose(output.asnumpy(), expected, rtol=1e-5, atol=1e-5)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("mode", [context.GRAPH_MODE, context.PYNATIVE_MODE])
def test_stft_empty_input_raises(mode):
    """
    Feature: STFT CPU input validation.
    Description: Run STFT with an empty 1D input and n_fft=64 in graph and pynative modes.
    Expectation: A ValueError is raised because n_fft exceeds the input length.
    """
    context.set_context(mode=mode, device_target="CPU")

    class STFTNet(nn.Cell):
        """Call the STFT primitive with fixed FFT settings."""

        def __init__(self):
            super().__init__()
            self.stft = ops.STFT(64, 16, 64, False, True, True)

        def construct(self, x, window):
            return self.stft(x, window)

    x = Tensor(np.empty((0,), dtype=np.float32))
    window = Tensor(np.ones((64,), dtype=np.float32))
    with pytest.raises(ValueError, match="n_fft"):
        STFTNet()(x, window)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_real_2d_numpy_reference():
    """
    Feature: STFT CPU kernel.
    Description: Test a 2D real input and default window against the NumPy reference.
    Expectation: The result matches the NumPy reference.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(0)
    x_np = np.random.randn(2, 128).astype(np.float32)
    _compare_stft_with_numpy(x_np, 16, 4, 16)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_normalized_numpy_reference():
    """
    Feature: STFT CPU kernel.
    Description: Test normalized STFT against the NumPy reference.
    Expectation: The result matches the NumPy reference.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(2)
    x_np = np.random.randn(3, 64).astype(np.float32)
    _compare_stft_with_numpy(x_np, 8, 2, 8, normalized=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_complex_input_numpy_reference():
    """
    Feature: STFT CPU kernel.
    Description: Test a complex input with twosided complex output against the NumPy reference.
    Expectation: The result matches the NumPy reference.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(3)
    real = np.random.randn(2, 64).astype(np.float32)
    imag = np.random.randn(2, 64).astype(np.float32)
    x_np = (real + 1j * imag).astype(np.complex64)
    _compare_stft_with_numpy(x_np, 8, 2, 8, onesided=False, return_complex=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_complex_window_numpy_reference():
    """
    Feature: STFT CPU kernel.
    Description: Test a real input and complex window against the NumPy reference.
    Expectation: The result matches the NumPy reference.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(4)
    x_np = np.random.randn(2, 64).astype(np.float32)
    window_np = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    _compare_stft_with_numpy(x_np, 8, 2, 8, window_np, onesided=False, return_complex=True)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_mixed_precision_numpy_reference():
    """
    Feature: STFT CPU kernel.
    Description: Test dtype promotion for a float32 input and float64 window.
    Expectation: The output has float64 precision and matches the NumPy reference.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(5)
    x_np = np.random.randn(2, 64).astype(np.float32)
    window_np = np.hanning(8).astype(np.float64)
    _compare_stft_with_numpy(x_np, 8, 2, 8, window_np)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level0',
          card_mark='onecard', essential_mark='essential')
def test_stft_onesided_with_complex_window_errors():
    """
    Feature: STFT CPU kernel.
    Description: Test onesided=True with a complex window.
    Expectation: A ValueError is raised.
    """
    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    np.random.seed(6)
    x_np = np.random.randn(2, 64).astype(np.float32)
    window_np = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    with pytest.raises(ValueError):
        F.stft(Tensor(x_np), 8, hop_length=2, win_length=8, window=Tensor(window_np), center=False,
               onesided=True, return_complex=True)
        _pynative_executor.sync()


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("data_type", [np.float32, np.float64])
def test_stft_real_input(data_type):
    """
    Feature: STFT cpu kernel real data input.
    Description: test the rightness of STFT cpu kernel.
    Expectation: Success.
    """
    x_np = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]]).astype(data_type)
    x_ms = Tensor(x_np)
    out_ms = F.stft(x_ms, 4, center=False, onesided=True)
    expect = np.array([[[[4, 0], [4, 0], [4, 0]],
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0]]],
                       [[[4, 0], [4, 0], [4, 0]],
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0]]]]).astype(data_type)
    assert np.allclose(out_ms.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("input_data_type", [np.complex64, np.complex128])
@pytest.mark.parametrize("win_data_type", [np.complex64, np.complex128])
def test_stft_complex_input(input_data_type, win_data_type):
    """
    Feature: STFT cpu kernel complex data input.
    Description: test the rightness of STFT cpu kernel.
    Expectation: Success.
    """
    x_np = np.array([[1j, 1j, 1j, 1j, 1j, 1j], [1j, 1j, 1j, 1j, 1j, 1j]]).astype(input_data_type)
    win_np = np.array([1j, 1j, 1j, 1j]).astype(win_data_type)
    x_ms = Tensor(x_np)
    win_ms = Tensor(win_np)
    out_ms = F.stft(x_ms, 4, window=win_ms, center=False, onesided=False, return_complex=True)
    expect = np.array([[[-4, -4, -4],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[-4, -4, -4],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]]).astype(np.complex128)
    assert np.allclose(out_ms.asnumpy(), expect)


@arg_mark(plat_marks=['cpu_linux', 'cpu_windows', 'cpu_macos'], level_mark='level1',
          card_mark='onecard', essential_mark='essential')
@pytest.mark.parametrize("input_data_type", [np.complex64, np.complex128])
@pytest.mark.parametrize("win_data_type", [np.float32, np.float64])
def test_stft_diff_type(input_data_type, win_data_type):
    """
    Feature: STFT cpu kernel.
    Description: test the rightness of STFT cpu kernel.
    Expectation: Success.
    """
    x_np = np.array([[1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j],
                     [1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j, 1 + 1j]]).astype(input_data_type)
    win_np = np.array([1, 1, 1, 1]).astype(win_data_type)
    x_ms = Tensor(x_np)
    win_ms = Tensor(win_np)
    out_ms = F.stft(x_ms, 4, window=win_ms, center=False, onesided=False,
                    return_complex=True, normalized=True)
    expect = np.array([[[2 + 2j, 2 + 2j, 2 + 2j],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]],
                       [[2 + 2j, 2 + 2j, 2 + 2j],
                        [0, 0, 0],
                        [0, 0, 0],
                        [0, 0, 0]]]).astype(np.complex128)
    assert np.allclose(out_ms.asnumpy(), expect)
