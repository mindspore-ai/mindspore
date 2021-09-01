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
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_func_contrast_eager():
    """ mindspore eager mode normal testcase:contrast op"""
    # Original waveform
    waveform = np.array([[1, 2], [3, 4]], dtype=np.float32)
    # Expect waveform
    expect_waveform = np.array([[1., -8.742277e-08],
                                [-1., 1.748455e-07]],
                               dtype=np.float32)
    contrast_op = audio.Contrast(75.0)
    # Filtered waveform by contrast
    output = contrast_op(waveform)
    count_unequal_element(expect_waveform, output, 0.0001, 0.0001)


def test_func_contrast_pipeline():
    """ mindspore pipeline mode normal testcase:contrast op"""
    # Original waveform
    waveform = np.array([[0.4941969, 0.53911686, 0.4846254], [0.10841596, 0.029320478, 0.52353495],
                         [0.23657, 0.087965, 0.43579]], dtype=np.float64)
    # Expect waveform
    expect_waveform = np.array([[7.032282948493957520e-01, 7.328570485115051270e-01, 6.967759728431701660e-01],
                                [2.311619222164154053e-01, 6.433061510324478149e-02, 7.226532697677612305e-01],
                                [4.539981484413146973e-01, 1.895205676555633545e-01, 6.622338891029357910e-01]],
                               dtype=np.float64)
    dataset = ds.NumpySlicesDataset(waveform, ["audio"], shuffle=False)
    contrast_op = audio.Contrast()
    # Filtered waveform by contrast
    dataset = dataset.map(input_columns=["audio"], operations=contrast_op, num_parallel_workers=8)
    i = 0
    for item in dataset.create_dict_iterator(output_numpy=True):
        count_unequal_element(expect_waveform[i, :], item['audio'], 0.0001, 0.0001)
        i += 1


def test_contrast_invalid_input():
    def test_invalid_input(test_name, enhancement_amount, error, error_msg):
        logger.info("Test Contrast with bad input: {0}".format(test_name))
        with pytest.raises(error) as error_info:
            audio.Contrast(enhancement_amount)
        assert error_msg in str(error_info.value)

    test_invalid_input("invalid enhancement_amount parameter type as a String", "75.0", TypeError,
                       "Argument enhancement_amount with value 75.0 is not of type [<class 'float'>, <class 'int'>],"
                       + " but got <class 'str'>.")
    test_invalid_input("invalid enhancement_amount parameter value", -1, ValueError,
                       "Input enhancement_amount is not within the required interval of [0, 100].")
    test_invalid_input("invalid enhancement_amount parameter value", 101, ValueError,
                       "Input enhancement_amount is not within the required interval of [0, 100].")


if __name__ == "__main__":
    test_func_contrast_eager()
    test_func_contrast_pipeline()
    test_contrast_invalid_input()
