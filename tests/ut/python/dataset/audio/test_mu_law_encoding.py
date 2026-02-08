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
"""Test MuLawEncoding."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio


def test_mu_law_encoding():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding in pipeline mode
    Expectation: The data is processed successfully
    """
    def gen():
        data = np.array([[0.1, 0.2, 0.3, 0.4]])
        yield (np.array(data, dtype=np.float32),)

    dataset = ds.GeneratorDataset(source=gen, column_names=["multi_dim_data"])

    dataset = dataset.map(operations=audio.MuLawEncoding(), input_columns=["multi_dim_data"])

    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert i["multi_dim_data"].shape == (1, 4)
        expected = np.array([[203, 218, 228, 234]])
        assert np.array_equal(i["multi_dim_data"], expected)


def test_mu_law_encoding_eager():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding in eager mode
    Expectation: The data is processed successfully
    """
    waveform = np.array([[0.1, 0.2, 0.3, 0.4]])
    output = audio.MuLawEncoding(128)(waveform)
    assert output.shape == (1, 4)
    expected = np.array([[98, 106, 111, 115]])
    assert np.array_equal(output, expected)


def test_mu_law_encoding_uncallable():
    """
    Feature: MuLawEncoding
    Description: Test param check of MuLawEncoding
    Expectation: Throw correct error and message
    """
    try:
        waveform = np.random.rand(2, 4)
        output = audio.MuLawEncoding(-3)(waveform)
        assert output.shape == (2, 4)
    except ValueError as e:
        assert 'Input quantization_channels is not within the required interval of [1, 2147483647].' in str(e)


def test_mu_law_encoding_and_decoding():
    """
    Feature: MuLawEncoding and MuLawDecoding
    Description: Test MuLawEncoding and MuLawDecoding in eager mode
    Expectation: The data is processed successfully
    """
    waveform = np.array([[98, 106, 111, 115]])
    output_decoding = audio.MuLawDecoding(128)(waveform)
    output_encoding = audio.MuLawEncoding(128)(output_decoding)
    assert np.array_equal(waveform, output_encoding)


def test_mu_law_encoding_transform():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.random.randn(4)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4,)

    # Test eager input dimension/shape
    waveform = np.random.randn(4, 5)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 5)

    # Test eager input dimension/shape
    waveform = np.random.randn(4, 4, 2)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 4, 2)

    # Test eager input dimension/shape
    waveform = np.random.randn(4, 5, 4, 2)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 5, 4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.float16)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.float32)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.float64)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.int16)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.int32)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)

    # Test eager data type
    waveform = np.random.randn(4, 2).astype(np.int64)
    mu_law_encoding = audio.MuLawEncoding()
    output = mu_law_encoding(waveform)
    assert np.shape(output) == (4, 2)


def test_mu_law_encoding_param_check():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    waveform = np.random.randn(4, 2).tolist()
    mu_law_encoding = audio.MuLawEncoding()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        mu_law_encoding(waveform)

    # Test with invalid value parameter (ValueError expected)
    with pytest.raises(ValueError) as error_info:
        err_msg = "Input quantization_channels is not within the required interval of [1, 2147483647]."
        audio.MuLawEncoding(-1)
    assert err_msg in str(error_info.value)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError) as error_info:
        err_msg = "Argument quantization_channels with value 3.0 is not of type [<class 'int'>], but got <class 'f" \
                  "loat'>."
        audio.MuLawEncoding(3.0)
    assert err_msg in str(error_info.value)

    # Test with invalid type parameter (TypeError expected)
    with pytest.raises(TypeError) as error_info:
        err_msg = "Argument quantization_channels with value 3 is not of type [<class 'int'>], but got <class 'str'>."
        audio.MuLawEncoding("3")
    assert err_msg in str(error_info.value)


if __name__ == "__main__":
    test_mu_law_encoding()
    test_mu_law_encoding_eager()
    test_mu_law_encoding_uncallable()
    test_mu_law_encoding_and_decoding()
    test_mu_law_encoding_transform()
    test_mu_law_encoding_param_check()
