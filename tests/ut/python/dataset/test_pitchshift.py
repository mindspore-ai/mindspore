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
"""
Testing PitchShift Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import WindowType


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_pitchshift_pipeline():
    """
    Feature: Test pipeline mode normal testcase: PitchShift op
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_PitchShift_pipeline")

    wav = [[[1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 1, 2, 3, 0, 1, 0, 2, 4, 5, 3, 1, 2, 3, 4]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.PitchShift(sample_rate=16000, n_steps=4, bins_per_octave=12, n_fft=16, win_length=16,
                           hop_length=4, window=WindowType.HANN)

    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["PitchShift"])
    result = np.array([[0.8897, 1.0983, 2.4355, 1.8842, 2.2082,
                        3.6461, 2.4232, 1.7691, 3.2835, 3.3354,
                        2.1773, 3.3544, 4.0488, 3.1631, 1.9124,
                        2.2346, 2.2417, 3.6008, 1.9539, 1.3373,
                        0.4311, 2.0768, 2.6538, 1.5035, 1.5668,
                        2.3749, 3.9702, 3.5922, 1.7618, 1.2730]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["PitchShift"], result, 0.0001, 0.0001)


def test_pitchshift_eager():
    """
    Feature: Mindspore eager mode normal testcase: pitchshift op
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_pitchshift_eager")
    wav = np.array([[[1, 1, 2, 3, 2, 3, 4, 5, 1, 2, 3, 4, 5, 2, 3, 2, 1, 2, 3, 0, 1, 0, 2, 4, 5, 3, 1, 2, 3, 4]]])
    out = audio.PitchShift(sample_rate=16000, n_steps=4, bins_per_octave=12, n_fft=16, win_length=16, hop_length=4,
                           window=WindowType.HANN)(wav)
    result = np.array([[[0.8897, 1.0983, 2.4355, 1.8842, 2.2082,
                         3.6461, 2.4232, 1.7691, 3.2835, 3.3354,
                         2.1773, 3.3544, 4.0488, 3.1631, 1.9124,
                         2.2346, 2.2417, 3.6008, 1.9539, 1.3373,
                         0.4311, 2.0768, 2.6538, 1.5035, 1.5668,
                         2.3749, 3.9702, 3.5922, 1.7618, 1.2730]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_pitchshift_param():
    """
    Feature: Test pitchshift invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        _ = audio.PitchShift(sample_rate="s", n_steps=4)
    except TypeError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)

    try:
        _ = audio.PitchShift(sample_rate=-1, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(n_steps="s", sample_rate=16)
    except TypeError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Argument n_steps with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.PitchShift(bins_per_octave=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input bins_per_octave is not within the required interval of [-2147483648, 0) and (0, 2147483647]." \
                in str(error)
    try:
        _ = audio.PitchShift(bins_per_octave="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Argument bins_per_octave  with value s is not of type [<class 'int'>], but got <class 'str'>." \
                in str(error)

    try:
        _ = audio.PitchShift(n_fft=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(n_fft=0, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(win_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(win_length="s", sample_rate=16, n_steps=4)
    except TypeError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.PitchShift(hop_length=-1, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(hop_length=-100, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.PitchShift(win_length=300, n_fft=200, sample_rate=16, n_steps=4)
    except ValueError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
    try:
        _ = audio.PitchShift(window=False, sample_rate=16, n_steps=4)
    except TypeError as error:
        logger.info("Got an exception in pitchshift: {}".format(str(error)))
        assert "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>." \
               in str(error)


if __name__ == "__main__":
    test_pitchshift_pipeline()
    test_pitchshift_eager()
    test_pitchshift_param()
