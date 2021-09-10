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
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as audio
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


def test_func_deemph_biquad_eager():
    """ mindspore eager mode normal testcase:deemph_biquad op"""
    # Original waveform
    waveform = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.04603508, 0.11216372, 0.19070681],
                                [0.18414031, 0.31054966, 0.42633607]], dtype=np.float64)
    deemph_biquad_op = audio.DeemphBiquad(44100)
    # Filtered waveform by deemphbiquad
    output = deemph_biquad_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_func_deemph_biquad_pipeline():
    """ mindspore pipeline mode normal testcase:deemph_biquad op"""
    # Original waveform
    waveform = np.array([[0.2, 0.2, 0.3], [0.4, 0.5, 0.7]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[0.0895, 0.1279, 0.1972],
                                [0.1791, 0.3006, 0.4583]], dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    deemph_biquad_op = audio.DeemphBiquad(48000)
    # Filtered waveform by deemphbiquad
    dataset = dataset.map(input_columns=["audio"], operations=deemph_biquad_op, num_parallel_workers=8)
    i = 0
    for data in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], data['audio'], 0.0001, 0.0001)
        i += 1


def test_invalid_input_all():
    waveform = np.random.rand(2, 1000)
    def test_invalid_input(test_name, sample_rate, error, error_msg):
        logger.info("Test DeemphBiquad with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.DeemphBiquad(sample_rate)(waveform)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid sample_rate parameter type as a float", 44100.5, TypeError,
                       "Argument sample_rate with value 44100.5 is not of type [<class 'int'>],"
                       + " but got <class 'float'>.")
    test_invalid_input("invalid sample_rate parameter type as a String", "44100", TypeError,
                       "Argument sample_rate with value 44100 is not of type [<class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid sample_rate parameter value", 45000, ValueError,
                       "Input sample_rate should be 44100 or 48000, but got 45000.")


if __name__ == '__main__':
    test_func_deemph_biquad_eager()
    test_func_deemph_biquad_pipeline()
    test_invalid_input_all()
