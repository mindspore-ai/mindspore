# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test melscale_fbanks."""

import numpy as np
import pytest

from mindspore.dataset import audio
from . import count_unequal_element


def test_melscale_fbanks_normal():
    """
    Feature: melscale_fbanks.
    Description: Test normal operation with NormType.NONE and MelType.HTK.
    Expectation: The output data is as expected.
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
    Expectation: The output data is as expected.
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
    Expectation: The output data is as expected.
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
    Expectation: The output data is as expected.
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

    def test_invalid_input(n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.melscale_fbanks(n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type)
        assert error_msg in str(error_info.value)

    test_invalid_input(99999999999, 0, 50, 5, 100, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "n_freqs")
    test_invalid_input(10.5, 0, 50, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "n_freqs")
    test_invalid_input(10, None, 50, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "f_min")
    test_invalid_input(10, 0, None, 5, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "f_max")
    test_invalid_input(10, 0, 50, 10.1, 100, audio.NormType.NONE, audio.MelType.HTK,
                       TypeError, "n_mels")
    test_invalid_input(20, 0, 50, 999999999999, 100, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "n_mels")
    test_invalid_input(10, 0, 50, 5, 100.1, audio.NormType.NONE,
                       audio.MelType.HTK, TypeError, "sample_rate")
    test_invalid_input(20, 0, 50, 5, 999999999999, audio.NormType.NONE,
                       audio.MelType.HTK, ValueError, "sample_rate")
    test_invalid_input(10, 0, 50, 5, 100, None, audio.MelType.HTK,
                       TypeError, "norm")
    test_invalid_input(10, 0, 50, 5, 100, audio.NormType.SLANEY, None,
                       TypeError, "mel_type")


def test_melscale_fbanks_transform():
    """
    Feature: MelscaleFbanks
    Description: Test MelscaleFbanks with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    norm = audio.NormType.NONE
    mel_type = audio.MelType.HTK
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, norm=norm, mel_type=mel_type)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32

    # test norm is NONE; mel_type is SLANEY
    norm = audio.NormType.NONE
    mel_type = audio.MelType.SLANEY
    output = audio.melscale_fbanks(100, 10, 50, 5, 100, norm=norm, mel_type=mel_type)
    assert output.shape == (100, 5)
    assert output.dtype == np.float32

    # test norm is SLANEY; mel_type is HTK
    norm = audio.NormType.SLANEY
    mel_type = audio.MelType.HTK
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, norm=norm, mel_type=mel_type)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32

    # test norm is SLANEY; mel_type is SLANEY
    norm = audio.NormType.SLANEY
    mel_type = audio.MelType.SLANEY
    output = audio.melscale_fbanks(8, 2, 50, 4, 100, norm=norm, mel_type=mel_type)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32

    # test default parameter
    output = audio.melscale_fbanks(8, 2, 50, 4, 100)
    assert output.shape == (8, 4)
    assert output.dtype == np.float32


def test_melscale_fbanks_param_check():
    """
    Feature: MelscaleFbanks
    Description: Test MelscaleFbanks with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    with pytest.raises(TypeError, match="Argument n_freqs with value .* is not of type .*int.*"
                                        "but got .* "):
        audio.melscale_fbanks(10.0, 2, 50, 4, 100)

    # Test n_freqs TypeError
    with pytest.raises(TypeError, match="Argument n_freqs with value .* is not of type .*int.*"
                                        "but got .* "):
        audio.melscale_fbanks('10', 2, 50, 4, 100)

    # Test n_freqs TypeError
    with pytest.raises(TypeError, match="Argument n_freqs with value .* is not of type .*int.*"
                                        "but got .* "):
        audio.melscale_fbanks(None, 2, 50, 4, 100)

    # Test n_freqs is 2147483648
    with pytest.raises(ValueError, match="Input n_freqs is not within the required"
                                         " interval of .*0, 2147483647.*"):
        audio.melscale_fbanks(2147483648, 2, 50, 4, 100)

    # Test n_freqs ValueError
    with pytest.raises(ValueError, match="Input n_freqs is not within the required"
                                         " interval of .*0, 2147483647.*"):
        audio.melscale_fbanks(-1, 2, 50, 4, 100)

    # Test f_min TypeError
    with pytest.raises(TypeError, match="Argument f_min with value .* is not of type .*int.*float.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 's', 50, 4, 100)

    # Test f_min ValueError
    with pytest.raises(ValueError, match="Input f_min is not within the "
                                         "required interval of .*0, 16777216.*"):
        audio.melscale_fbanks(10, -1, 50, 4, 100)

    # Test f_min ValueError
    with pytest.raises(ValueError, match="Input f_min should be no more than f_max, "
                                         "but got f_min.*and f_max.*"):
        audio.melscale_fbanks(10, 10.001, 10.0, 4, 100)

    # Test f_max TypeError
    with pytest.raises(TypeError, match="Argument f_max with value .* is not of type .*int.*float.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 2, '50', 4, 100)

    # Test f_max ValueError
    with pytest.raises(ValueError, match="Input f_max is not within the "
                                         "required interval of .*0, 16777216.*"):
        audio.melscale_fbanks(10, 2, -50, 4, 100)

    # Test n_mels TypeError
    with pytest.raises(TypeError, match="Argument n_mels with value .* is not of type .*int.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 2, 50, True, 100)

    # Test n_mels ValueError
    with pytest.raises(ValueError, match="Input n_mels is not within the "
                                         "required interval of .*1, 2147483647.*"):
        audio.melscale_fbanks(10, 2, 50, -1, 100)

    # Test n_mels is 2147483648
    with pytest.raises(ValueError, match="Input n_mels is not within the "
                                         "required interval of .*1, 2147483647.*"):
        audio.melscale_fbanks(10, 2, 50, 2147483648, 100)

    # Test sample_rate TypeError
    with pytest.raises(TypeError, match="Argument sample_rate with value .* is not of type .*int.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 2, 50, 4, [1, 100])

    # Test sample_rate ValueError
    with pytest.raises(ValueError, match="Input sample_rate is not within the "
                                         "required interval of .*1, 2147483647.*"):
        audio.melscale_fbanks(10, 2, 50, 4, 0)

    # Test norm TypeError
    with pytest.raises(TypeError, match="Argument norm with value .* is not of type .*enum 'NormType'.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 2, 50, 4, 100, 1)

    # Test mel_type TypeError
    with pytest.raises(TypeError, match="Argument mel_type with value .* is not of type .*enum 'MelType'.*"
                                        "but got .* "):
        audio.melscale_fbanks(10, 2, 50, 4, 100, audio.NormType.NONE, 'htk')


if __name__ == "__main__":
    test_melscale_fbanks_normal()
    test_melscale_fbanks_none_slaney()
    test_melscale_fbanks_with_slaney_htk()
    test_melscale_fbanks_with_slaney_slaney()
    test_melscale_fbanks_invalid_input()
    test_melscale_fbanks_transform()
    test_melscale_fbanks_param_check()
