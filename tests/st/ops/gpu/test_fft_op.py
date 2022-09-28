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

import pytest
import numpy as np
import mindspore as ms
import mindspore.nn as nn

from mindspore import Tensor
from mindspore.ops.operations import math_ops

ms.context.set_context(mode=ms.context.PYNATIVE_MODE, device_target="GPU")


LOSS_R_32 = 1e-4
LOSS_R_64 = 1e-5
LOSS_C_32 = 2e-4
LOSS_C_64 = 2e-5


class FFTWithSizeNet(nn.Cell):
    def __init__(self, signal_ndim, inverse, real, norm="backward", onesided=True, signal_sizes=()):
        super(FFTWithSizeNet, self).__init__()
        self.fft = math_ops.FFTWithSize(signal_ndim, inverse, real, norm, onesided, signal_sizes)

    def construct(self, x):
        return self.fft(x)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_fft_1d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 1d signals as the input of fft1
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.fft(np_x, norm=norm)

    fft = FFTWithSizeNet(signal_ndim=1, inverse=False, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_fft_2d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 2d signals as the input of fft2
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.fft2(np_x, norm=norm)

    fft = FFTWithSizeNet(signal_ndim=2, inverse=False, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_fft_3d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 3d signals as the input of fft3
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4, 4, 4) + 1j * np.random.rand(4, 4, 4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.fftn(np_x, axes=(-3, -2, -1), norm=norm)

    fft = FFTWithSizeNet(signal_ndim=3, inverse=False, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_ifft_1d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 1d frequencies as the input of ifft1
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.ifft(np_x, norm=norm)

    fft = FFTWithSizeNet(signal_ndim=1, inverse=True, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_ifft_2d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 2d frequencies as the input of ifft2
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.ifft2(np_x, norm=norm)

    fft = FFTWithSizeNet(signal_ndim=2, inverse=True, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_ifft_3d(dynamic, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 3d frequencies as the input of ifft3
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4, 4, 4) + 1j * np.random.rand(4, 4, 4, 4)
    np_x = np_x.astype(dtype_c)
    np_y = np.fft.ifftn(np_x, axes=(-3, -2, -1), norm=norm)

    fft = FFTWithSizeNet(signal_ndim=3, inverse=True, real=False, norm=norm)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)

    print(f"max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("rank", [1, 2, 3])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_fft_ifft_circle_call(dynamic, rank, norm, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of n_d signals as the input of ifft_n(fft_n(*))
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    np_x = np.random.rand(4, 4, 4) + 1j * np.random.rand(4, 4, 4)
    np_x = np_x.astype(dtype_c)
    fft = FFTWithSizeNet(signal_ndim=rank, inverse=False, real=False, norm=norm)
    ifft = FFTWithSizeNet(signal_ndim=rank, inverse=True, real=False, norm=norm)
    ms_x = ms.Tensor(np_x)
    ms_y = ifft(fft(ms_x))

    print(f"max error: {np.max(np.abs(ms_x.asnumpy() - ms_y.asnumpy()))}")
    if dtype_r == np.float32:
        atol = LOSS_C_32
    elif dtype_r == np.float64:
        atol = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(ms_x.asnumpy(), ms_y.asnumpy(), atol=atol)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("pass_signal_sizes", [True, False])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_rfft_1d_bidrection(dynamic, norm, onesided, pass_signal_sizes, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 1d signals as the input of irfft1(rfft1(*))
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    # We cannot use random number as the input of irfft, which is a kind of undefined behavior.
    # It has to be the output of rfft.
    np_x = np.random.rand(4, 4)
    np_x = np_x.astype(dtype_r)
    if onesided:
        np_y = np.fft.rfft(np_x, norm=norm)
        np_x_recovered = np.fft.irfft(np_y, len(np_x), norm=norm)
    else:
        np_y = np.fft.fft(np_x, norm=norm)
        np_x_recovered = np.fft.ifft(np_y, norm=norm)
        np_x_recovered = np.real(np_x_recovered)

    fft = FFTWithSizeNet(signal_ndim=1, inverse=False, real=True, norm=norm, onesided=onesided)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)
    signal_sizes = ms_x.shape[-1:] if pass_signal_sizes else ()  # shape of signal must be odd.
    ifft = FFTWithSizeNet(signal_ndim=1, inverse=True, real=True,
                          norm=norm, onesided=onesided, signal_sizes=signal_sizes)
    if dynamic:
        y_dyn_shape = [None for _ in range(ms_y.ndim)]
        y_dyn = Tensor(shape=y_dyn_shape, dtype=ms_y.dtype)
        ifft.set_inputs(y_dyn)
    ms_x_recovered = ifft(ms_y)

    print(f"rfft max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}", end=" ")
    print(f"irfft max error: {np.max(np.abs(np_x_recovered - ms_x_recovered.asnumpy()))}", end=" ")
    print(f"recover max error: {np.max(np.abs(ms_x.asnumpy() - ms_x_recovered.asnumpy()))}")
    if dtype_r == np.float32:
        atol_r = LOSS_R_32
        atol_c = LOSS_C_32
    elif dtype_r == np.float64:
        atol_r = LOSS_R_64
        atol_c = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol_c)
    assert np.allclose(np_x_recovered, ms_x_recovered.asnumpy(), atol=atol_r)
    assert np.allclose(ms_x.asnumpy(), ms_x_recovered.asnumpy(), atol=atol_r)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("pass_signal_sizes", [True, False])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_rfft_2d_bidrection(dynamic, norm, onesided, pass_signal_sizes, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 2d signals as the input of irfft2(rfft2(*))
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    # We cannot use random number as the input of irfft, which is a kind of undefined behavior.
    # It has to be the output of rfft.
    np_x = np.random.rand(4, 4, 4)
    np_x = np_x.astype(dtype_r)
    if onesided:
        np_y = np.fft.rfft2(np_x, norm=norm)
        np_x_recovered = np.fft.irfft2(np_y, np_x.shape[-2:], norm=norm)
    else:
        np_y = np.fft.fft2(np_x, norm=norm)
        np_x_recovered = np.fft.ifft2(np_y, norm=norm)
        np_x_recovered = np.real(np_x_recovered)

    fft = FFTWithSizeNet(signal_ndim=2, inverse=False, real=True, norm=norm, onesided=onesided)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)
    signal_sizes = ms_x.shape[-2:] if pass_signal_sizes else ()
    ifft = FFTWithSizeNet(signal_ndim=2, inverse=True, real=True,
                          norm=norm, onesided=onesided, signal_sizes=signal_sizes)
    if dynamic:
        y_dyn_shape = [None for _ in range(ms_y.ndim)]
        y_dyn = Tensor(shape=y_dyn_shape, dtype=ms_y.dtype)
        ifft.set_inputs(y_dyn)
    ms_x_recovered = ifft(ms_y)

    print(f"rfft max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}", end=" ")
    print(f"irfft max error: {np.max(np.abs(np_x_recovered - ms_x_recovered.asnumpy()))}", end=" ")
    print(f"recover max error: {np.max(np.abs(ms_x.asnumpy() - ms_x_recovered.asnumpy()))}")
    if dtype_r == np.float32:
        atol_r = LOSS_R_32
        atol_c = LOSS_C_32
    elif dtype_r == np.float64:
        atol_r = LOSS_R_64
        atol_c = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol_c)
    assert np.allclose(np_x_recovered, ms_x_recovered.asnumpy(), atol=atol_r)
    assert np.allclose(ms_x.asnumpy(), ms_x_recovered.asnumpy(), atol=atol_r)


@pytest.mark.level1
@pytest.mark.env_onecard
@pytest.mark.platform_x86_gpu_training
@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("norm", ["forward", "backward", "ortho"])
@pytest.mark.parametrize("onesided", [True, False])
@pytest.mark.parametrize("pass_signal_sizes", [True, False])
@pytest.mark.parametrize("dtype_r, dtype_c", [(np.float32, np.complex64), (np.float64, np.complex128)])
def test_rfft_3d_bidrection(dynamic, norm, onesided, pass_signal_sizes, dtype_r, dtype_c):
    """
    Feature: operator FFTWithSize
    Description: batch of 3d signals as the input of irfft3(rfft3(*))
    Expectation: success or throw AssertionError exception or raise TypeError.
    """
    # We cannot use random number as the input of irfft, which is a kind of undefined behavior.
    # It has to be the output of rfft.
    np_x = np.random.rand(4, 4, 4, 4)
    np_x = np_x.astype(dtype_r)
    if onesided:
        np_y = np.fft.rfftn(np_x, axes=(-3, -2, -1), norm=norm)
        np_x_recovered = np.fft.irfftn(np_y, np_x.shape[-3:], axes=(-3, -2, -1), norm=norm)
    else:
        np_y = np.fft.fftn(np_x, axes=(-3, -2, -1), norm=norm)
        np_x_recovered = np.fft.ifftn(np_y, axes=(-3, -2, -1), norm=norm)
        np_x_recovered = np.real(np_x_recovered)

    fft = FFTWithSizeNet(signal_ndim=3, inverse=False, real=True, norm=norm, onesided=onesided)
    ms_x = Tensor(np_x)
    if dynamic:
        x_dyn_shape = [None for _ in range(ms_x.ndim)]
        x_dyn = Tensor(shape=x_dyn_shape, dtype=ms_x.dtype)
        fft.set_inputs(x_dyn)
    ms_y = fft(ms_x)
    signal_sizes = ms_x.shape[-3:] if pass_signal_sizes else ()
    ifft = FFTWithSizeNet(signal_ndim=3, inverse=True, real=True,
                          norm=norm, onesided=onesided, signal_sizes=signal_sizes)
    if dynamic:
        y_dyn_shape = [None for _ in range(ms_y.ndim)]
        y_dyn = Tensor(shape=y_dyn_shape, dtype=ms_y.dtype)
        ifft.set_inputs(y_dyn)
    ms_x_recovered = ifft(ms_y)

    print(f"rfft max error: {np.max(np.abs(np_y - ms_y.asnumpy()))}", end=" ")
    print(f"irfft max error: {np.max(np.abs(np_x_recovered - ms_x_recovered.asnumpy()))}", end=" ")
    print(f"recover max error: {np.max(np.abs(ms_x.asnumpy() - ms_x_recovered.asnumpy()))}")
    if dtype_r == np.float32:
        atol_r = LOSS_R_32
        atol_c = LOSS_C_32
    elif dtype_r == np.float64:
        atol_r = LOSS_R_64
        atol_c = LOSS_C_64
    else:
        raise TypeError("Only support float32 or float64!")
    assert np.allclose(np_y, ms_y.asnumpy(), atol=atol_c)
    assert np.allclose(np_x_recovered, ms_x_recovered.asnumpy(), atol=atol_r)
    assert np.allclose(ms_x.asnumpy(), ms_x_recovered.asnumpy(), atol=atol_r)


if __name__ == '__main__':
    for p_net in [False, True]:
        for p_norm in ["forward", "backward", "ortho"]:
            for p_dtype_r, p_dtype_c in [(np.float32, np.complex64), (np.float64, np.complex128)]:
                print(f"\n[CASE] net: {p_net}, norm: {p_norm}, dtype: {p_dtype_r}")
                print("  [fft1]", end=" ")
                test_fft_1d(p_net, p_norm, p_dtype_r, p_dtype_c)
                print("  [fft2]", end=" ")
                test_fft_2d(p_net, p_norm, p_dtype_r, p_dtype_c)
                print("  [fft3]", end=" ")
                test_fft_3d(p_net, p_norm, p_dtype_r, p_dtype_c)
                print("  [ifft1]", end=" ")
                test_ifft_1d(p_net, p_norm, p_dtype_r, p_dtype_c)
                print("  [ifft2]", end=" ")
                test_ifft_2d(p_net, p_norm, p_dtype_r, p_dtype_c)
                print("  [ifft3]", end=" ")
                test_ifft_3d(p_net, p_norm, p_dtype_r, p_dtype_c)
                for p_rank in range(1, 4):
                    print(f"  [fft_ifft_{p_rank}]", end=" ")
                    test_fft_ifft_circle_call(p_net, p_rank, p_norm, p_dtype_r, p_dtype_c)
                for p_onesided in [True, False]:
                    for p_pass_sizes in [True, False]:
                        print(f"  [rfft_irfft_1](onesided={p_onesided}, pass_sizes={p_pass_sizes})", end=" ")
                        test_rfft_1d_bidrection(p_net, p_norm, p_onesided, p_pass_sizes, p_dtype_r, p_dtype_c)
                        print(f"  [rfft_irfft_2](onesided={p_onesided}, pass_sizes={p_pass_sizes})", end=" ")
                        test_rfft_2d_bidrection(p_net, p_norm, p_onesided, p_pass_sizes, p_dtype_r, p_dtype_c)
                        print(f"  [rfft_irfft_3](onesided={p_onesided}, pass_sizes={p_pass_sizes})", end=" ")
                        test_rfft_3d_bidrection(p_net, p_norm, p_onesided, p_pass_sizes, p_dtype_r, p_dtype_c)
