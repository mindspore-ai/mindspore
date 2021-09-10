# Copyright 2021 Huawei Technologies Co., Ltd
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
Testing MuLawDecoding op in DE.
"""

import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
from mindspore import log as logger


def test_mu_law_decoding():
    """
    Test mu_law_decoding_op (pipeline).
    """
    logger.info("Test MuLawDecoding.")

    def gen():
        data = np.array([[10, 100, 70, 200]])
        yield (np.array(data, dtype=np.float32),)

    dataset = ds.GeneratorDataset(source=gen, column_names=["multi_dim_data"])

    dataset = dataset.map(operations=audio.MuLawDecoding(), input_columns=["multi_dim_data"])

    for i in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        assert i["multi_dim_data"].shape == (1, 4)
        expected = np.array([[-0.6459359526634216, -0.009046762250363827, -0.04388953000307083, 0.08788024634122849]])
        assert np.array_equal(i["multi_dim_data"], expected)

    logger.info("Finish testing MuLawDecoding.")


def test_mu_law_decoding_eager():
    """
    Test mu_law_decoding_op callable (eager).
    """
    logger.info("Test MuLawDecoding callable.")

    input_t = np.array([70, 170])
    output_t = audio.MuLawDecoding()(input_t)
    assert output_t.shape == (2,)
    excepted = np.array([-0.04388953000307083, 0.02097884565591812])
    assert np.array_equal(output_t, excepted)

    logger.info("Finish testing MuLawDecoding.")


def test_mu_law_decoding_uncallable():
    """
    Test mu_law_decoding_op not callable.
    """
    logger.info("Test MuLawDecoding not callable.")

    try:
        input_t = np.random.rand(2, 4)
        output_t = audio.MuLawDecoding(-3)(input_t)
        assert output_t.shape == (2, 4)
    except ValueError as e:
        assert 'Input quantization_channels is not within the required interval of [1, 2147483647].' in str(e)

    logger.info("Finish testing MuLawDecoding.")


if __name__ == "__main__":
    test_mu_law_decoding()
    test_mu_law_decoding_eager()
    test_mu_law_decoding_uncallable()
