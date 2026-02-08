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
"""Test MuLawDecoding."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio


def test_mu_law_decoding():
    """
    Feature: MuLawDecoding
    Description: Test MuLawDecoding in pipeline mode
    Expectation: Output is the same as expected output
    """
    def gen():
        data = np.array([[10, 100, 70, 200]])
        yield (np.array(data, dtype=np.float32),)

    dataset = ds.GeneratorDataset(source=gen, column_names=["multi_dim_data"])

    dataset = dataset.map(operations=audio.MuLawDecoding(), input_columns=["multi_dim_data"])

    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert i["multi_dim_data"].shape == (1, 4)
        expected = np.array([[-0.6459359526634216, -0.009046762250363827, -0.04388953000307083, 0.08788024634122849]])
        assert np.array_equal(i["multi_dim_data"], expected)


def test_mu_law_decoding_eager():
    """
    Feature: MuLawDecoding
    Description: Test MuLawDecoding in eager mode
    Expectation: Output is the same as expected output
    """
    waveform = np.array([70, 170])
    output = audio.MuLawDecoding(128)(waveform)
    assert output.shape == (2,)
    excepted = np.array([0.00506480922922492, 26.928272247314453])
    assert np.array_equal(output, excepted)


def test_mu_law_decoding_uncallable():
    """
    Feature: MuLawDecoding
    Description: Test parameter check of MuLawDecoding
    Expectation: Error is raised as expected
    """
    try:
        waveform = np.random.rand(2, 4)
        output = audio.MuLawDecoding(-3)(waveform)
        assert output.shape == (2, 4)
    except ValueError as e:
        assert 'Input quantization_channels is not within the required interval of [1, 2147483647].' in str(e)


def test_mu_law_decoding_transform():
    """
    Feature: MuLawDecoding
    Description: Test MuLawDecoding with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    waveform = np.random.randn(4)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test with various parameter combinations
    waveform = np.random.randn(4, 5)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test with various parameter combinations
    waveform = np.random.randn(4, 4, 2)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of float16 type
    waveform = np.random.randn(4, 5, 4, 2)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of float32 type
    waveform = np.random.randn(4, 2).astype(np.float16)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of float64 type
    waveform = np.random.randn(4, 2).astype(np.float32)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of float64 type
    waveform = np.random.randn(4, 2).astype(np.float64)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of int32 type
    waveform = np.random.randn(4, 2).astype(np.int16)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of int64 type
    waveform = np.random.randn(4, 2).astype(np.int32)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)

    # Test of int64 type
    waveform = np.random.randn(4, 2).astype(np.int64)
    mu_law_decoding = audio.MuLawDecoding()
    mu_law_decoding(waveform)


def test_mu_law_decoding_param_check():
    """
    Feature: MuLawDecoding
    Description: Test MuLawDecoding with invalid input data types and parameters
    Expectation: Correct error types and messages are raised as expected
    """

    waveform = np.random.randn(4, 2).tolist()
    mu_law_decoding = audio.MuLawDecoding()
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        mu_law_decoding(waveform)

    # Test with invalid value parameter (ValueError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(ValueError,
                       match="Input quantization_channels is not within the required interval of \\[1, 2147483647\\]."):
        mu_law_decoding = audio.MuLawDecoding(-256)
        mu_law_decoding(waveform)

    # Test with invalid value parameter (ValueError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(ValueError,
                       match="Input quantization_channels is not within the required interval of \\[1, 2147483647\\]."):
        mu_law_decoding = audio.MuLawDecoding(2147483648)
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(ValueError,
                       match="Input quantization_channels is not within the required interval of \\[1, 2147483647\\]."):
        mu_law_decoding = audio.MuLawDecoding(0)
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value 8.0 is not of type \\[<class 'int'>\\], "
                             "but got <class 'float'>."):
        mu_law_decoding = audio.MuLawDecoding(8.0)
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value \\(1, 2\\) is not of type \\[<class 'int'>\\], "
                             "but got <class 'tuple'>."):
        mu_law_decoding = audio.MuLawDecoding((1, 2))
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value \\[1, 2\\] is not of type \\[<class 'int'>\\], "
                             "but got <class 'list'>."):
        mu_law_decoding = audio.MuLawDecoding([1, 2])
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value \\{1, 2\\} is not of type \\[<class 'int'>\\], "
                             "but got <class 'set'>."):
        mu_law_decoding = audio.MuLawDecoding({1, 2})
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value \\{1: 1, 2: 2\\} is not of type \\[<class "
                             "'int'>\\], but got <class 'dict'>."):
        mu_law_decoding = audio.MuLawDecoding({1: 1, 2: 2})
        mu_law_decoding(waveform)

    # Test with invalid type parameter (TypeError expected)
    waveform = np.random.randn(4, 2).astype(np.int32)
    with pytest.raises(TypeError,
                       match="Argument quantization_channels with value a is not of type \\[<class "
                             "'int'>\\], but got <class 'str'>."):
        mu_law_decoding = audio.MuLawDecoding('a')
        mu_law_decoding(waveform)


if __name__ == "__main__":
    test_mu_law_decoding()
    test_mu_law_decoding_eager()
    test_mu_law_decoding_uncallable()
    test_mu_law_decoding_transform()
    test_mu_law_decoding_param_check()
