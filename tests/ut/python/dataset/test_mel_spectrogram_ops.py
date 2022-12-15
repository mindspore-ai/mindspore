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
Testing MelSpectrogram Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import WindowType, BorderType, NormType, MelType


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_melspectrogram_pipeline():
    """
    Feature: Test pipeline mode normal testcase: MelSpectrogram op
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_melspectrogram_pipeline")

    wav = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.MelSpectrogram(sample_rate=16000, n_fft=16, win_length=16, hop_length=8, f_min=0.0, f_max=10000.0,
                               pad=0, n_mels=8, window=WindowType.HANN, power=2.0, normalized=False, center=True,
                               pad_mode=BorderType.REFLECT, onesided=True, norm=NormType.NONE, mel_scale=MelType.HTK)
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["MelSpectrogram"])
    result = np.array([[[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                        [1.6105e+00, 2.8416e+01, 4.0224e+00, 1.4698e+01],
                        [1.8027e+01, 3.1808e+02, 4.5026e+01, 1.6452e+02],
                        [7.9213e+00, 8.4180e+00, 5.6739e+00, 2.2122e+00],
                        [6.0452e+00, 6.5609e+00, 4.5775e+00, 1.8347e+00],
                        [5.6763e-01, 9.4627e-01, 6.4849e-01, 3.0038e-01],
                        [3.1647e-01, 1.2753e+00, 7.9531e-01, 1.7264e-01],
                        [2.6995e+00, 2.0453e+00, 2.6940e+00, 3.5556e+00]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MelSpectrogram"], result, 0.0001, 0.0001)


def test_melspectrogram_eager():
    """
    Feature: Test eager mode normal testcase: MelSpectrogram op
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_melspectrogram_eager")
    wav = np.array([[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]])
    out = audio.MelSpectrogram(sample_rate=16000, n_fft=16, win_length=16, hop_length=8, f_min=0.0, f_max=5000.0,
                               pad=0, n_mels=8, window=WindowType.HANN, power=2.0, normalized=False, center=True,
                               pad_mode=BorderType.REFLECT, onesided=True, norm=NormType.NONE,
                               mel_scale=MelType.HTK)(wav)
    result = np.array([[[[0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00],
                         [4.1355e+00, 7.2968e+01, 1.0329e+01, 3.7741e+01],
                         [1.5502e+01, 2.7353e+02, 3.8720e+01, 1.4148e+02],
                         [3.0792e+00, 3.2723e+00, 2.2056e+00, 8.5993e-01],
                         [1.0531e+01, 1.1192e+01, 7.5435e+00, 2.9411e+00],
                         [5.6983e-01, 8.2424e-01, 8.0414e-01, 3.9367e-01],
                         [3.7583e-01, 6.6152e-01, 3.4257e-01, 1.6248e-01]]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_melspectrogram_param():
    """
    Feature: Test melspectrogram invalid parameter
    Description: Test some invalid parameters
    Expectation: throw ValueError, TypeError or RuntimeError exception
    """
    try:
        _ = audio.MelSpectrogram(sample_rate=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(sample_rate="s")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MelSpectrogram(f_max=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input f_max is not within the required interval of [0, 16777216]." in str(error)
    try:
        _ = audio.MelSpectrogram(f_min=-1.0)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input f_min is not within the required interval of (0, 16777216]." in str(error)
    try:
        _ = audio.MelSpectrogram(norm=-1)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument norm with value -1 is not of type [<enum 'NormType'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MelSpectrogram(mel_scale=-1)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument mel_type with value -1 is not of type [<enum 'MelType'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MelSpectrogram(n_fft=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(n_fft=0)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(win_length=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(win_length="s")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MelSpectrogram(hop_length=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(hop_length=-100)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(win_length=300, n_fft=200)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(pad=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input pad is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MelSpectrogram(power=-1)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Input power is not within the required interval of (0, 16777216]." in str(error)
    try:
        _ = audio.MelSpectrogram(n_fft=False)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(error)
    try:
        _ = audio.MelSpectrogram(n_fft="s")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(window=False)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(pad_mode=False)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(onesided="LianLinghang")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument onesided with value LianLinghang is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(center="XiaDanni")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument center with value XiaDanni is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MelSpectrogram(normalized="s")
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MelSpectrogram(normalized=1)
    except TypeError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MelSpectrogram(f_max=1.0, f_min=2.0, sample_rate=16000)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "f_max should be greater than or equal to f_min, but got f_min: 2.0 and f_max: 1.0." in str(error)
    try:
        _ = audio.MelSpectrogram(f_min=60.0, f_max=None, sample_rate=100)
    except ValueError as error:
        logger.info("Got an exception in MelSpectrogram: {}".format(str(error)))
        assert "MelSpectrogram: sample_rate // 2 should be greater than f_min when f_max is set to None, "\
               "but got f_min: 60.0." in str(error)


if __name__ == "__main__":
    test_melspectrogram_pipeline()
    test_melspectrogram_eager()
    test_melspectrogram_param()
