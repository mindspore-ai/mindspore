# Copyright 2022 Huawei Technologies Co., Ltd :
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
Testing MFCC Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import WindowType, BorderType, MelType, NormType, NormMode


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_mfcc_pipeline():
    """
    Feature: Mindspore pipeline mode normal testcase: mfcc op
    Description: Input audio signal to test pipeline
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_mfcc_pipeline")

    wav = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.MFCC(sample_rate=16000, n_mfcc=4, dct_type=2, norm=NormMode.ORTHO, log_mels=True,
                     melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0,
                                "f_max": 10000.0, "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0,
                                "normalized": False, "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                "norm": NormType.NONE, "mel_scale": MelType.HTK})
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["MFCC"])
    result = np.array([[[2.7625, 5.6919, 3.6229, 3.9756],
                        [0.8142, 3.2698, 1.4946, 3.0683],
                        [-1.6855, -0.8312, -1.1395, 0.0481],
                        [-2.1808, -2.5489, -2.3110, -3.1485]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["MFCC"], result, 0.0001, 0.0001)


def test_mfcc_eager():
    """
    Feature: Mindspore eager mode normal testcase: mfcc op
    Description: Input audio signal to test eager
    Expectation: Generate expected output after cases were executed
    """
    logger.info("test_mfcc_eager")
    wav = np.array([[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]])
    out = audio.MFCC(sample_rate=16000, n_mfcc=4, dct_type=2, norm=NormMode.ORTHO, log_mels=True,
                     melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": False,
                                "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                "norm": NormType.NONE, "mel_scale": MelType.HTK})(wav)
    result = np.array([[[[2.7625, 5.6919, 3.6229, 3.9756],
                         [0.8142, 3.2698, 1.4946, 3.0683],
                         [-1.6855, -0.8312, -1.1395, 0.0481],
                         [-2.1808, -2.5489, -2.3110, -3.1485]]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_mfcc_param():
    """
    Feature: Test mfcc invalid parameter.
    Description: Test some invalid parameters.
    Expectation: throw ValueError, TypeError or RuntimeError exception.
    """
    try:
        _ = audio.MFCC(sample_rate=-1)
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(log_mels=-1)
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument log_mels with value -1 is not of type [<class 'bool'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MFCC(norm="Karl Marx")
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument norm with value Karl Marx is not of type [<enum 'NormMode'>], but got <class 'str'>." \
        in str(error)
    try:
        _ = audio.MFCC(dct_type=-1)
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "dct_type must be 2, but got : -1." in str(error)
    try:
        _ = audio.MFCC(sample_rate=-1)
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(sample_rate="s")
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument sample_rate with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": -1,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input f_max is not within the required interval of (0, 16777216]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": -1, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input n_mels should be greater than or equal to n_mfcc, but got n_mfcc: 40 and n_mels: 5." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": -1, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument norm with value -1 is not of type [<enum 'NormType'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": -1})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument mel_type with value -1 is not of type [<enum 'MelType'>], but got <class 'int'>." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": -1, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 0, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 0, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 50, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": "s", "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": -1, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 200, "win_length": 300, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 50, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": -1, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input pad is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": -1, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except ValueError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Input power is not within the required interval of [0, 16777216]." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": "XiaDanni", "win_length": 16, "hop_length": 8, "f_min": 0.0,
                                  "f_max": 10000.0, "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0,
                                  "normalized": True, "center": True, "pad_mode": BorderType.REFLECT,
                                  "onesided": True, "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument n_fft with value XiaDanni is not of type [<class 'int'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": False, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": True,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": False, "onesided": True, "norm": NormType.NONE,
                                  "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": "LianLinghang",
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument onesided with value LianLinghang is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": True,
                                  "center": "XiaDanni", "pad_mode": BorderType.REFLECT, "onesided": False,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument center with value XiaDanni is not of type [<class 'bool'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": "s",
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": False,
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.MFCC(melkwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "f_min": 0.0, "f_max": 10000.0,
                                  "pad": 0, "n_mels": 5, "window": WindowType.HANN, "power": 2.0, "normalized": 1,
                                  "center": True, "pad_mode": BorderType.REFLECT, "onesided": "LianLinghang",
                                  "norm": NormType.NONE, "mel_scale": MelType.HTK})
    except TypeError as error:
        logger.info("Got an exception in MFCC: {}".format(str(error)))
        assert "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>." in str(error)


if __name__ == '__main__':
    test_mfcc_pipeline()
    test_mfcc_eager()
    test_mfcc_param()
