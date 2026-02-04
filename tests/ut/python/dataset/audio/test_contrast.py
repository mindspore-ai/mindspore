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
"""Test Contrast."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from . import count_unequal_element


def test_contrast_eager():
    """
    Feature: Contrast
    Description: Test Contrast in eager mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array(
        [[1.0, -8.742277e-08], [-1.0, 1.748455e-07]], dtype=np.float32
    )
    contrast = audio.Contrast(75.0)
    # Filtered waveform by contrast
    output = contrast(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_contrast_pipeline():
    """
    Feature: Contrast
    Description: Test Contrast in pipeline mode with valid input
    Expectation: Output is equal to the expected output
    """
    # Original waveform
    waveform = np.array(
        [
            [0.4941969, 0.53911686, 0.4846254],
            [0.10841596, 0.029320478, 0.52353495],
            [0.23657, 0.087965, 0.43579],
        ],
        dtype=np.float64,
    )
    # Expect waveform
    expect_waveform = np.array(
        [
            [
                7.032282948493957520e-01,
                7.328570485115051270e-01,
                6.967759728431701660e-01,
            ],
            [
                2.311619222164154053e-01,
                6.433061510324478149e-02,
                7.226532697677612305e-01,
            ],
            [
                4.539981484413146973e-01,
                1.895205676555633545e-01,
                6.622338891029357910e-01,
            ],
        ],
        dtype=np.float64,
    )
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    contrast = audio.Contrast()
    # Filtered waveform by contrast
    dataset = dataset.map(
        input_columns=["audio"], operations=contrast, num_parallel_workers=8
    )
    i = 0
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item["audio"], 0.0001, 0.0001)
        i += 1


def test_contrast_invalid_input():
    """
    Feature: Contrast
    Description: Test Contrast with invalid input
    Expectation: Correct error and message are thrown as expected
    """

    def test_invalid_input(enhancement_amount, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Contrast(enhancement_amount)
        assert error_msg in str(error_info.value)

    test_invalid_input(
        "75.0",
        TypeError,
        "Argument enhancement_amount with value 75.0 is not of type [<class 'float'>, <class 'int'>],"
        + " but got <class 'str'>.",
    )
    test_invalid_input(
        -1,
        ValueError,
        "Input enhancement_amount is not within the required interval of [0, 100].",
    )
    test_invalid_input(
        101,
        ValueError,
        "Input enhancement_amount is not within the required interval of [0, 100].",
    )


def test_contrast_transform():
    """
    Feature: Contrast
    Description: Test Contrast with various valid input parameters and data types
    Expectation: The operation completes successfully and output values match expectations
    """
    # test contrast pipeline functions
    x = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.float32)
    data = x.reshape((1, 2, x.size // 2))
    dataset = ds.NumpySlicesDataset(data, column_names=["audio"], shuffle=False)
    contrast = audio.Contrast(75.0)
    dataset = dataset.map(input_columns=["audio"], operations=contrast)
    expect = np.array(
        [
            [
                0.39793506264686584,
                0.634295642375946,
                0.7071067690849304,
                0.21418575942516327,
            ],
            [
                0.5365509390830994,
                0.7071067690849304,
                0.8438679575920105,
                0.39793506264686584,
            ],
        ]
    )
    for outdata in dataset.create_dict_iterator():
        assert (outdata["audio"].asnumpy() == expect).all()

    # test contrast normal,test data type
    x = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.float16)
    data = x.reshape((1, 2, x.size // 2))
    dataset = ds.NumpySlicesDataset(data, column_names=["audio"], shuffle=False)
    contrast = audio.Contrast(75.0)
    dataset = dataset.map(input_columns=["audio"], operations=contrast)
    expect = np.array(
        [
            [
                0.3978559970855713,
                0.6342154145240784,
                0.7071067690849304,
                0.21413616836071014,
            ],
            [
                0.5366076231002808,
                0.7071067690849304,
                0.8440122008323669,
                0.3978559970855713,
            ],
        ]
    )
    for outdata in dataset.create_dict_iterator():
        assert (outdata["audio"].asnumpy() == expect).all()

    # test contrast normal,test data type
    x = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.float64)
    data = x.reshape((1, 2, x.size // 2))
    dataset = ds.NumpySlicesDataset(data, column_names=["audio"], shuffle=False)
    contrast = audio.Contrast(75.0)
    dataset = dataset.map(input_columns=["audio"], operations=contrast)
    expect = np.array(
        [
            [
                0.39793506041510684,
                0.63429561925971,
                0.7071067811865475,
                0.21418574965896425,
            ],
            [
                0.5365509230253295,
                0.7071067811865475,
                0.8438679440532546,
                0.39793506041510684,
            ],
        ]
    )
    for outdata in dataset.create_dict_iterator():
        np.testing.assert_allclose(
            outdata["audio"].asnumpy(), expect, rtol=1e-16, atol=1e-16
        )

    # test contrast normal,test data type
    x = np.array([[2, 4, 5, 1], [3, 5, 7, 2]], dtype=np.int32)
    data = x.reshape((1, 2, x.size // 2))
    dataset = ds.NumpySlicesDataset(data, column_names=["audio"], shuffle=False)
    contrast = audio.Contrast(75.0)
    dataset = dataset.map(input_columns=["audio"], operations=contrast)
    expect = np.array(
        [
            [-8.742277657347586e-08, 1.7484555314695172e-07, 1.0, 1.0],
            [-1.0, 1.0, -1.0, -8.742277657347586e-08],
        ]
    )
    for outdata in dataset.create_dict_iterator():
        assert (outdata["audio"].asnumpy() == expect).all()

    # Test eager
    waveform = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.float32)
    contrast = audio.Contrast(2.0)
    output = contrast(waveform)
    expect = np.array(
        [
            [
                0.311428040266037,
                0.5890526175498962,
                0.7071067690849304,
                0.15798240900039673,
            ],
            [
                0.45624876022338867,
                0.7071067690849304,
                0.8898522257804871,
                0.311428040266037,
            ],
        ]
    )
    assert (output == expect).all()

    # Test eager
    waveform = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.float32)
    # enhancement_amount default 75.0
    contrast = audio.Contrast()
    output = contrast(waveform)
    expect = np.array(
        [
            [
                0.39793506264686584,
                0.634295642375946,
                0.7071067690849304,
                0.21418575942516327,
            ],
            [
                0.5365509390830994,
                0.7071067690849304,
                0.8438679575920105,
                0.39793506264686584,
            ],
        ]
    )
    assert (output == expect).all()


def test_contrast_param_check():
    """
    Feature: Contrast
    Description: Test Contrast with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """
    # test contrast pipeline exception function
    x = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.bytes_)
    data = x.reshape((1, 2, x.size // 2))
    dataset = ds.NumpySlicesDataset(data, ["audio"], shuffle=False)
    contrast = audio.Contrast(75.0)
    dataset = dataset.map(input_columns=["audio"], operations=contrast)
    with pytest.raises(
        RuntimeError,
        match="map operation: \\[Contrast\\] failed. Contrast: "
        "the data type of input tensor does not match the requirement of operator. "
        "Expecting tensor in type of \\[int, float, double\\].",
    ):
        for _ in dataset.create_dict_iterator():
            pass

    # Test eager
    waveform = np.array([[0.2, 0.4, 0.5, 0.1], [0.3, 0.5, 0.7, 0.2]], dtype=np.bytes_)
    contrast = audio.Contrast(75.0)
    with pytest.raises(
        RuntimeError,
        match="Contrast: the data type of input tensor does not match the requirement of "
        "operator. Expecting tensor in type of \\[int, float, double\\].",
    ):
        contrast(waveform)

    # test contrast invalid input
    def invalid_input(enhancement_amount, error, error_msg):
        with pytest.raises(error) as error_info:
            audio.Contrast(enhancement_amount)
        assert error_msg in str(error_info.value)

    invalid_input(
        "75.0",
        TypeError,
        "Argument enhancement_amount with value 75.0 is not of type [<class 'float'>, <class 'int'>],"
        + " but got <class 'str'>.",
    )
    invalid_input(
        -1,
        ValueError,
        "Input enhancement_amount is not within the required interval of [0, 100].",
    )
    invalid_input(
        101,
        ValueError,
        "Input enhancement_amount is not within the required interval of [0, 100].",
    )


if __name__ == "__main__":
    test_contrast_eager()
    test_contrast_pipeline()
    test_contrast_invalid_input()
    test_contrast_transform()
    test_contrast_param_check()
