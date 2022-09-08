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

import numpy as np
import pytest
from mindspore import Tensor
from mindspore.ops import functional as F


def np_all_close_with_loss(out, expect):
    """np_all_close_with_loss"""
    return np.allclose(out, expect, 0.00001, 0.00001, equal_nan=True)


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("data_type", [np.float, np.double])
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
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


@pytest.mark.level0
@pytest.mark.platform_x86_cpu
@pytest.mark.env_onecard
@pytest.mark.parametrize("input_data_type", [np.complex64, np.complex128])
@pytest.mark.parametrize("win_data_type", [np.float, np.double])
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
