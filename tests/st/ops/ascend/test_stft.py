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
import numpy as np
import pytest

import mindspore as ms
import mindspore.context as context
from mindspore import Tensor
from mindspore import ops
from tests.mark_utils import arg_mark


def _run_stft(x_np, n_fft, hop_length=None, win_length=None, window_np=None,
              center=False, normalized=False, onesided=None, return_complex=None):
    x_ms = Tensor(x_np)
    window_ms = Tensor(window_np) if window_np is not None else None
    out = ops.stft(x_ms, n_fft, hop_length=hop_length, win_length=win_length,
                   window=window_ms, center=center, normalized=normalized,
                   onesided=onesided, return_complex=return_complex)
    return out.asnumpy()


def _stft_compare(x_np, n_fft, **kwargs):
    """Run stft on CPU (legacy primitive) then on Ascend (aclnn pyboost) and compare."""
    context.set_context(device_target="CPU")
    cpu_out = _run_stft(x_np, n_fft, **kwargs)

    context.set_context(device_target="Ascend")
    ascend_out = _run_stft(x_np, n_fft, **kwargs)

    assert cpu_out.shape == ascend_out.shape, \
        f"shape mismatch: cpu={cpu_out.shape} ascend={ascend_out.shape}"
    assert cpu_out.dtype == ascend_out.dtype, \
        f"dtype mismatch: cpu={cpu_out.dtype} ascend={ascend_out.dtype}"
    np.testing.assert_allclose(ascend_out, cpu_out, rtol=1e-4, atol=1e-4)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
@pytest.mark.parametrize('dtype', [np.float32])
def test_stft_real_2d(mode, dtype):
    """
    Feature: aclnn STFT on Ascend.
    Description: 2D real input, default window, onesided real output, compared against CPU STFT.
    Expectation: Ascend output matches CPU output.
    """
    context.set_context(mode=mode)
    np.random.seed(0)
    x_np = np.random.randn(2, 128).astype(dtype)
    _stft_compare(x_np, n_fft=16, hop_length=4, win_length=16,
                  onesided=True, return_complex=False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_real_1d_with_window(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: 1D real input with hann-like window, twosided complex output, compared against CPU.
    Expectation: Ascend output matches CPU output.
    """
    context.set_context(mode=mode)
    np.random.seed(1)
    x_np = np.random.randn(96).astype(np.float32)
    n_fft = 16
    win_np = np.hanning(n_fft).astype(np.float32)
    _stft_compare(x_np, n_fft=n_fft, hop_length=4, win_length=n_fft, window_np=win_np,
                  onesided=False, return_complex=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_normalized(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: 2D real input, normalized=True, real output, compared against CPU STFT.
    Expectation: Ascend output matches CPU output.
    """
    context.set_context(mode=mode)
    np.random.seed(2)
    x_np = np.random.randn(3, 64).astype(np.float32)
    _stft_compare(x_np, n_fft=8, hop_length=2, win_length=8,
                  normalized=True, onesided=True, return_complex=False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_complex_input(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: 2D complex64 input, return_complex=True, twosided, compared against CPU.
    Expectation: Ascend output matches CPU output.
    """
    context.set_context(mode=mode)
    np.random.seed(3)
    real = np.random.randn(2, 64).astype(np.float32)
    imag = np.random.randn(2, 64).astype(np.float32)
    x_np = (real + 1j * imag).astype(np.complex64)
    _stft_compare(x_np, n_fft=8, hop_length=2, win_length=8,
                  onesided=False, return_complex=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_real_input_complex_window(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: real input + complex window must promote to complex (not demote window to real).
    Expectation: Ascend output matches CPU output dtype-for-dtype.
    """
    context.set_context(mode=mode)
    np.random.seed(4)
    x_np = np.random.randn(2, 64).astype(np.float32)
    win_np = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    _stft_compare(x_np, n_fft=8, hop_length=2, win_length=8, window_np=win_np,
                  onesided=False, return_complex=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_mixed_precision(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: float32 input + float64 window must promote to float64 (not demote window).
    Expectation: Ascend output matches CPU output, both at float64 precision.
    """
    context.set_context(mode=mode)
    np.random.seed(5)
    x_np = np.random.randn(2, 64).astype(np.float32)
    win_np = np.hanning(8).astype(np.float64)
    _stft_compare(x_np, n_fft=8, hop_length=2, win_length=8, window_np=win_np,
                  onesided=True, return_complex=False)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_onesided_with_complex_window_errors(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: explicit onesided=True with a complex window must error (matches CPU semantics).
    Expectation: ValueError raised.
    """
    context.set_context(mode=mode, device_target="Ascend")
    np.random.seed(6)
    x_np = np.random.randn(2, 64).astype(np.float32)
    win_np = (np.random.randn(8) + 1j * np.random.randn(8)).astype(np.complex64)
    with pytest.raises(ValueError):
        _run_stft(x_np, n_fft=8, hop_length=2, win_length=8, window_np=win_np,
                  onesided=True, return_complex=True)


@arg_mark(plat_marks=['platform_ascend'], level_mark='level1', card_mark='onecard',
          essential_mark='essential')
@pytest.mark.parametrize('mode', [context.PYNATIVE_MODE])
def test_stft_known_constant(mode):
    """
    Feature: aclnn STFT on Ascend.
    Description: 2D constant ones input, verifies a deterministic spectrum on Ascend.
    Expectation: Ascend output matches the analytical expectation (sum_window at DC, zeros elsewhere).
    """
    context.set_context(mode=mode, device_target="Ascend")
    n_fft = 4
    x_np = np.ones((2, 6), dtype=np.float32)
    win_np = np.ones((n_fft,), dtype=np.float32)
    out = _run_stft(x_np, n_fft=n_fft, hop_length=1, win_length=n_fft, window_np=win_np,
                    onesided=True, return_complex=False)
    expect = np.array([[[[4, 0], [4, 0], [4, 0]],
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0]]],
                       [[[4, 0], [4, 0], [4, 0]],
                        [[0, 0], [0, 0], [0, 0]],
                        [[0, 0], [0, 0], [0, 0]]]], dtype=np.float32)
    np.testing.assert_allclose(out, expect, rtol=1e-4, atol=1e-4)
