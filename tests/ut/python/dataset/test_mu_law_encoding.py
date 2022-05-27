# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""
Testing MuLawEncoding op in DE.
"""

import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger


def test_mu_law_encoding():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding in pipeline mode
    Expectation: The data is processed successfully
    """
    logger.info("Test MuLawEncoding.")

    def gen():
        data = np.array([[0.1, 0.2, 0.3, 0.4]])
        yield (np.array(data, dtype=np.float32),)

    dataset = ds.GeneratorDataset(source=gen, column_names=["multi_dim_data"])

    dataset = dataset.map(operations=audio.MuLawEncoding(), input_columns=["multi_dim_data"])

    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert i["multi_dim_data"].shape == (1, 4)
        expected = np.array([[203, 218, 228, 234]])
        assert np.array_equal(i["multi_dim_data"], expected)

    logger.info("Finish testing MuLawEncoding.")


def test_mu_law_encoding_eager():
    """
    Feature: MuLawEncoding
    Description: Test MuLawEncoding in eager mode
    Expectation: The data is processed successfully
    """
    logger.info("Test MuLawEncoding callable.")

    input_t = np.array([[0.1, 0.2, 0.3, 0.4]])
    output_t = audio.MuLawEncoding(128)(input_t)
    assert output_t.shape == (1, 4)
    expected = np.array([[98, 106, 111, 115]])
    assert np.array_equal(output_t, expected)

    logger.info("Finish testing MuLawEncoding.")


def test_mu_law_encoding_uncallable():
    """
    Feature: MuLawEncoding
    Description: Test param check of MuLawEncoding
    Expectation: Throw correct error and message
    """
    logger.info("Test MuLawEncoding not callable.")

    try:
        input_t = np.random.rand(2, 4)
        output_t = audio.MuLawEncoding(-3)(input_t)
        assert output_t.shape == (2, 4)
    except ValueError as e:
        assert 'Input quantization_channels is not within the required interval of [1, 2147483647].' in str(e)

    logger.info("Finish testing MuLawEncoding.")


def test_mu_law_encoding_and_decoding():
    """
    Feature: MuLawEncoding and MuLawDecoding
    Description: Test MuLawEncoding and MuLawDecoding in eager mode
    Expectation: The data is processed successfully
    """
    logger.info("Test MuLawEncoding and MuLawDecoding callable.")

    input_t = np.array([[98, 106, 111, 115]])
    output_decoding = audio.MuLawDecoding(128)(input_t)
    output_encoding = audio.MuLawEncoding(128)(output_decoding)
    assert np.array_equal(input_t, output_encoding)

    logger.info("Finish testing MuLawEncoding and MuLawDecoding callable.")


if __name__ == "__main__":
    test_mu_law_encoding()
    test_mu_law_encoding_eager()
    test_mu_law_encoding_uncallable()
    test_mu_law_encoding_and_decoding()
