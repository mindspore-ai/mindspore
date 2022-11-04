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
# ==============================================================================
import numpy as np
import pytest

import mindspore.dataset.audio.utils as audio
from mindspore import log as logger


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def test_melscale_fbanks_normal():
    """
    Feature: melscale_fbanks.
    Description: Test normal operation with NormType.NONE and MelType.HTK.
    Expectation: The output data is the same as the result of torchaudio.functional.melscale_fbanks.
    """
    expect = np.array([[0.0000, 0.0000, 0.0000, 0.0000],
                       [0.5502, 0.0000, 0.0000, 0.0000],
                       [0.6898, 0.3102, 0.0000, 0.0000],
                       [0.0000, 0.9366, 0.0634, 0.0000],
                       [0.0000, 0.1924, 0.8076, 0.0000],
                       [0.0000, 0.0000, 0.4555, 0.5445],
                       [0.0000, 0.0000, 0.0000, 0.7247],
                       [0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, audio.NormType.NONE, audio.MelType.HTK)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_melscale_fbanks_none_slaney():
    """
    Feature: melscale_fbanks.
    Description: Test normal operation with NormType.NONE and MelType.SLANEY.
    Expectation: The output data is the same as the result of torchaudio.functional.melscale_fbanks.
    """
    expect = np.array([[0.0000, 0.0000, 0.0000, 0.0000],
                       [0.5357, 0.0000, 0.0000, 0.0000],
                       [0.7202, 0.2798, 0.0000, 0.0000],
                       [0.0000, 0.9762, 0.0238, 0.0000],
                       [0.0000, 0.2321, 0.7679, 0.0000],
                       [0.0000, 0.0000, 0.4881, 0.5119],
                       [0.0000, 0.0000, 0.0000, 0.7440],
                       [0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, audio.NormType.NONE, audio.MelType.SLANEY)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_melscale_fbanks_with_slaney_htk():
    """
    Feature: melscale_fbanks.
    Description: Test normal operation with NormType.SLANEY and MelType.HTK.
    Expectation: The output data is the same as the result of torchaudio.functional.melscale_fbanks.
    """
    output = audio.melscale_fbanks(10, 0, 50, 5, 100, audio.NormType.SLANEY, audio.MelType.HTK)
    expect = np.array([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                       [0.0843, 0.0000, 0.0000, 0.0000, 0.0000],
                       [0.0776, 0.0447, 0.0000, 0.0000, 0.0000],
                       [0.0000, 0.1158, 0.0055, 0.0000, 0.0000],
                       [0.0000, 0.0344, 0.0860, 0.0000, 0.0000],
                       [0.0000, 0.0000, 0.0741, 0.0454, 0.0000],
                       [0.0000, 0.0000, 0.0000, 0.1133, 0.0053],
                       [0.0000, 0.0000, 0.0000, 0.0355, 0.0822],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0760],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_melscale_fbanks_with_slaney_slaney():
    """
    Feature: melscale_fbanks.
    Description: Test normal operation with NormType.SLANEY and MelType.SLANEY.
    Expectation: The output data is the same as the result of torchaudio.functional.melscale_fbanks.
    """
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, audio.NormType.SLANEY, audio.MelType.SLANEY)
    expect = np.array([[0.0000, 0.0000, 0.0000, 0.0000],
                       [0.0558, 0.0000, 0.0000, 0.0000],
                       [0.0750, 0.0291, 0.0000, 0.0000],
                       [0.0000, 0.1017, 0.0025, 0.0000],
                       [0.0000, 0.0242, 0.0800, 0.0000],
                       [0.0000, 0.0000, 0.0508, 0.0533],
                       [0.0000, 0.0000, 0.0000, 0.0775],
                       [0.0000, 0.0000, 0.0000, 0.0000]], dtype=np.float64)
    count_unequal_element(expect, output, 0.0001, 0.0001)


def test_melscale_fbanks_invalid_input():
    """
    Feature: melscale_fbanks.
    Description: Test operation with invalid input.
    Expectation: Throw exception as expected.
    """

    def test_invalid_input(test_name, n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type, error, error_msg):
        logger.info("Test melscale_fbanks with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid n_freqs parameter Value", 99999999999, 0, 50, 5, 100, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "n_freqs")
    test_invalid_input("invalid n_freqs parameter type", 10.5, 0, 50, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "n_freqs")
    test_invalid_input("invalid f_min parameter type", 10, None, 50, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "f_min")
    test_invalid_input("invalid f_max parameter type", 10, 0, None, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "f_max")
    test_invalid_input("invalid n_mels parameter type", 10, 0, 50, 10.1, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "n_mels")
    test_invalid_input("invalid n_mels parameter Value", 20, 0, 50, 999999999999, 100, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "n_mels")
    test_invalid_input("invalid sample_rate parameter type", 10, 0, 50, 5, 100.1, audio.NormType.NONE,
                       audio.MelType.HTK, TypeError, "sample_rate")
    test_invalid_input("invalid sample_rate parameter Value", 20, 0, 50, 5, 999999999999, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "sample_rate")
    test_invalid_input("invalid norm parameter type", 10, 0, 50, 5, 100, None, audio.MelType.HTK,
                       TypeError, "norm")
    test_invalid_input("invalid norm parameter type", 10, 0, 50, 5, 100, audio.NormType.SLANEY, None,
                       TypeError, "mel_type")


if __name__ == "__main__":
    test_melscale_fbanks_normal()
    test_melscale_fbanks_none_slaney()
    test_melscale_fbanks_with_slaney_htk()
    test_melscale_fbanks_with_slaney_slaney()
    test_melscale_fbanks_invalid_input()
