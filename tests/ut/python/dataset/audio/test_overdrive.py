# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""Test Overdrive."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_overdrive_eager():
    """
    Feature: Overdrive
    Description: Test Overdrive in eager mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[1.47, 4.722, 5.863], [0.492, 0.235, 0.56]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[1., 1., 1.],
                                [0.74600005, 0.615, 0.77501255]], dtype=np.float32)
    overdrive = audio.Overdrive()
    # Filtered waveform by overdrive
    output = overdrive(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_overdrive_pipeline():
    """
    Feature: Overdrive
    Description: Test Overdrive in pipeline mode
    Expectation: The results are as expected
    """
    # Original waveform
    waveform = np.array([[0.1, 0.2], [0.4, 2.6]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[0.29598799, 0.52081579],
                                [0.7, 1.]], dtype=np.float32)
    dataset = ds.NumpySlicesDataset(waveform, ["waveform"], shuffle=False)
    overdrive = audio.Overdrive(10.0, 5.0)
    # Filtered waveform by overdrive
    dataset = dataset.map(
        input_columns=["waveform"], operations=overdrive)
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :],
                              item['waveform'], 0.0001, 0.0001)
        i += 1


def test_overdrive_invalid_input():
    """
    Feature: Overdrive
    Description: Test invalid parameter of Overdrive
    Expectation: Catch exceptions correctly
    """
    def test_invalid_input(gain, color, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Overdrive(gain, color)
        assert error_msg in str(error_info.value)

    test_invalid_input("20", 20, TypeError,
                       "Argument gain with value 20 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(10, "5", TypeError,
                       "Argument color with value 5 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input(100.23, 5.0, ValueError,
                       "Input gain is not within the required interval of [0, 100].")
    test_invalid_input(30, -0.333, ValueError,
                       "Input color is not within the required interval of [0, 100].")


if __name__ == "__main__":
    test_overdrive_eager()
    test_overdrive_pipeline()
    test_overdrive_invalid_input()
