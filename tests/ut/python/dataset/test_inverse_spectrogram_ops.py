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
Testing InverseSpectrogram Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import WindowType, BorderType


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_inverse_spectrogram_pipeline():
    """
    Feature: Test pipeline mode normal testcase: inversespectrogram op
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_inverse_spectrogram_pipeline")

    wav = [[[[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]], [[5.0, 5.0]],
             [[6.0, 6.0]], [[5.0, 5.0]], [[4.0, 4.0]], [[3.0, 3.0]]],
            [[[2.0, 2.0]], [[1.0, 1.0]], [[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]],
             [[3.0, 3.0]], [[4.0, 4.0]], [[5.0, 5.0]], [[6.0, 6.0]]]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.InverseSpectrogram(length=1, n_fft=16, win_length=16, hop_length=8, pad=0,
                                   window=WindowType.HANN, normalized=False, center=True,
                                   pad_mode=BorderType.REFLECT, onesided=True)
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["InverseSpectrogram"])
    result = np.array([[-0.1250], [0.0000]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["InverseSpectrogram"], result, 0.0001, 0.0001)


def test_inverse_spectrogram_eager():
    """
    Feature: Test pipeline mode normal testcase: inversespectrogram op
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_inverse_spectrogram_eager")
    wav = np.array([[[[1.0, 1.0]], [[2.0, 2.0]], [[3.0, 3.0]], [[4.0, 4.0]], [[5.0, 5.0]],
                     [[6.0, 6.0]], [[5.0, 5.0]], [[4.0, 4.0]], [[3.0, 3.0]]],
                    [[[2.0, 2.0]], [[1.0, 1.0]], [[0.0, 0.0]], [[1.0, 1.0]], [[2.0, 2.0]],
                     [[3.0, 3.0]], [[4.0, 4.0]], [[5.0, 5.0]], [[6.0, 6.0]]]])
    out = audio.InverseSpectrogram(length=1, n_fft=16, win_length=16, hop_length=8, pad=1,
                                   window=WindowType.HANN, normalized=False, center=True,
                                   pad_mode=BorderType.REFLECT, onesided=True)(wav)
    result = np.array([[0.1399], [0.1034]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_inversespectrogram_param():
    """
    Feature: Test inversespectrogram invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        _ = audio.InverseSpectrogram(length=-1)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input length is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, n_fft=-1)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, n_fft=0)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, win_length=-1)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, win_length="s")
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, hop_length=-1)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, hop_length=-100)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, win_length=300, n_fft=200)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, pad=-1)
    except ValueError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Input pad is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, n_fft=False)
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, n_fft="s")
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, window=False)
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, pad_mode=False)
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, onesided="GanJisong")
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument onesided with value GanJisong is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, center="MindSpore")
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument center with value MindSpore is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, normalized="s")
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.InverseSpectrogram(length=1, normalized=1)
    except TypeError as error:
        logger.info("Got an exception in InverseSpectrogram: {}".format(str(error)))
        assert "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>." in str(error)


if __name__ == "__main__":
    test_inverse_spectrogram_pipeline()
    test_inverse_spectrogram_eager()
    test_inversespectrogram_param()
