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
Testing Spectrogram Python API
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


def test_spectrogram_pipeline():
    """
    Feature: Mindspore pipeline mode normal testcase: spectrogram op.
    Description: Input audio signal to test pipeline.
    Expectation: Success.
    """
    logger.info("test_spectrogram_pipeline")

    wav = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.Spectrogram(n_fft=8)
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["Spectrogram"])
    result = np.array([[[2.8015e+01, 1.2100e+02, 3.1354e+02, 1.6900e+02, 2.5000e+01,
                         1.0843e+01, 1.2100e+02, 3.3150e+02],
                        [3.2145e+00, 3.3914e+01, 9.4728e+01, 4.5914e+01, 9.9142e+00,
                         4.5858e+00, 3.3914e+01, 9.5685e+01],
                        [1.0000e+00, 1.7157e-01, 1.5000e+00, 1.7157e-01, 1.7157e-01,
                         5.0000e-01, 1.7157e-01, 7.5000e-01],
                        [4.2893e-02, 2.5736e-01, 5.8579e-01, 2.5736e-01, 2.5736e-01,
                         5.8579e-01, 2.5736e-01, 1.2868e-01],
                        [5.0000e-01, 1.0000e+00, 8.5787e-02, 1.0000e+00, 1.0000e+00,
                         5.0000e-01, 1.0000e+00, 6.2868e-01]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["Spectrogram"], result, 0.0001, 0.0001)


def test_spectrogram_eager():
    """
    Feature: Mindspore eager mode normal testcase: spectrogram op.
    Description: Input audio signal to test eager.
    Expectation: Success.
    """
    logger.info("test_spectrogram_eager")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, win_length=8, window=WindowType.HANN,
                            pad_mode=BorderType.REFLECT)(np.array(wav, dtype="float"))
    result = np.array([[[2.8015e+01, 1.2100e+02, 3.1354e+02, 1.6900e+02, 2.5000e+01,
                         1.0843e+01, 1.2100e+02, 3.3150e+02],
                        [3.2145e+00, 3.3914e+01, 9.4728e+01, 4.5914e+01, 9.9142e+00,
                         4.5858e+00, 3.3914e+01, 9.5685e+01],
                        [1.0000e+00, 1.7157e-01, 1.5000e+00, 1.7157e-01, 1.7157e-01,
                         5.0000e-01, 1.7157e-01, 7.5000e-01],
                        [4.2893e-02, 2.5736e-01, 5.8579e-01, 2.5736e-01, 2.5736e-01,
                         5.8579e-01, 2.5736e-01, 1.2868e-01],
                        [5.0000e-01, 1.0000e+00, 8.5787e-02, 1.0000e+00, 1.0000e+00,
                         5.0000e-01, 1.0000e+00, 6.2868e-01]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_window_hamming_padmode_constant():
    """
    Feature: Test spectrogram parameter: window, pad_mode.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_window_hamming_padmode_constant")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, window=WindowType.HAMMING,
                            pad_mode=BorderType.CONSTANT)(np.array(wav, dtype="float"))
    result = np.array([[[1.1389e+01, 1.3736e+02, 3.5534e+02, 2.0164e+02, 3.0914e+01,
                         1.3465e+01, 1.3736e+02, 2.5064e+02],
                        [5.6934e+00, 3.1860e+01, 8.5291e+01, 3.8484e+01, 9.1907e+00,
                         4.5576e+00, 3.1860e+01, 1.0027e+02],
                        [1.9633e-01, 7.4475e-02, 1.2696e+00, 7.4475e-02, 7.4475e-02,
                         4.2320e-01, 7.4475e-02, 1.1765e+01],
                        [5.0570e-01, 3.8456e-01, 6.8326e-01, 3.8456e-01, 3.8456e-01,
                         6.8326e-01, 3.8456e-01, 5.4501e+00],
                        [6.1665e-01, 8.4640e-01, 7.2610e-02, 8.4640e-01, 8.4640e-01,
                         4.2320e-01, 8.4640e-01, 1.0642e+00]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_nfft_10_window_bartlett_padmode_edge():
    """
    Feature: Test spectrogram parameter: n_fft, window, pad_mode.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_nfft_10_window_bartlett_padmode_edge")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=10, window=WindowType.BARTLETT,
                            pad_mode=BorderType.EDGE)(np.array(wav, dtype="float"))
    result = np.array([[[4.0960e+01, 2.6244e+02, 4.0000e+02, 7.7440e+01, 2.5000e+01,
                         2.6244e+02, 5.9536e+02],
                        [4.7655e+00, 5.6721e+01, 9.4681e+01, 2.3822e+01, 5.9597e+00,
                         5.6721e+01, 1.1597e+02],
                        [5.3889e-01, 5.8360e-03, 9.2361e-01, 5.8359e-03, 9.2361e-01,
                         5.8360e-03, 2.4944e-01],
                        [1.1449e-01, 9.9859e-01, 3.1897e-01, 2.9828e-01, 1.0403e+00,
                         9.9859e-01, 1.3072e+00],
                        [1.8111e-01, 2.7416e-01, 4.7639e-01, 2.7416e-01, 4.7639e-01,
                         2.7416e-01, 7.0557e-02],
                        [6.4000e-01, 3.6000e-01, 0.0000e+00, 2.5600e+00, 1.0000e+00,
                         3.6000e-01, 1.4400e+00]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_onsided_false():
    """
    Feature: Test spectrogram parameter: onesided.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_onsided_false")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=10, window=WindowType.BARTLETT,
                            pad_mode=BorderType.EDGE, onesided=False)(np.array(wav, dtype="float"))
    result = np.array([[[4.0960e+01, 2.6244e+02, 4.0000e+02, 7.7440e+01, 2.5000e+01,
                         2.6244e+02, 5.9536e+02],
                        [4.7655e+00, 5.6721e+01, 9.4681e+01, 2.3822e+01, 5.9597e+00,
                         5.6721e+01, 1.1597e+02],
                        [5.3889e-01, 5.8360e-03, 9.2361e-01, 5.8359e-03, 9.2361e-01,
                         5.8360e-03, 2.4944e-01],
                        [1.1449e-01, 9.9859e-01, 3.1897e-01, 2.9828e-01, 1.0403e+00,
                         9.9859e-01, 1.3072e+00],
                        [1.8111e-01, 2.7416e-01, 4.7639e-01, 2.7416e-01, 4.7639e-01,
                         2.7416e-01, 7.0557e-02],
                        [6.4000e-01, 3.6000e-01, 0.0000e+00, 2.5600e+00, 1.0000e+00,
                         3.6000e-01, 1.4400e+00],
                        [1.8111e-01, 2.7416e-01, 4.7639e-01, 2.7416e-01, 4.7639e-01,
                         2.7416e-01, 7.0557e-02],
                        [1.1449e-01, 9.9859e-01, 3.1897e-01, 2.9828e-01, 1.0403e+00,
                         9.9859e-01, 1.3072e+00],
                        [5.3889e-01, 5.8360e-03, 9.2361e-01, 5.8359e-03, 9.2361e-01,
                         5.8360e-03, 2.4944e-01],
                        [4.7655e+00, 5.6721e+01, 9.4681e+01, 2.3822e+01, 5.9597e+00,
                         5.6721e+01, 1.1597e+02]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_power_0():
    """
    Feature: Test spectrogram parameter: power.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_power_0")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, window=WindowType.HANN,
                            pad_mode=BorderType.REFLECT, power=0)(np.array(wav, dtype="float"))
    result = np.array([[[[5.2929e+00, 0.0000e+00],
                         [1.1000e+01, 0.0000e+00],
                         [1.7707e+01, 0.0000e+00],
                         [1.3000e+01, 0.0000e+00],
                         [5.0000e+00, 0.0000e+00],
                         [3.2929e+00, 0.0000e+00],
                         [1.1000e+01, 0.0000e+00],
                         [1.8207e+01, 0.0000e+00]],
                        [[-1.7929e+00, -2.5288e-07],
                         [-5.5000e+00, 1.9142e+00],
                         [-9.7071e+00, 7.0711e-01],
                         [-6.5000e+00, -1.9142e+00],
                         [-2.5000e+00, -1.9142e+00],
                         [-1.2929e+00, 1.7071e+00],
                         [-5.5000e+00, 1.9142e+00],
                         [-9.7071e+00, 1.2071e+00]],
                        [[-1.0000e+00, 0.0000e+00],
                         [0.0000e+00, -4.1421e-01],
                         [1.0000e+00, -7.0711e-01],
                         [0.0000e+00, 4.1421e-01],
                         [0.0000e+00, 4.1421e-01],
                         [0.0000e+00, -7.0711e-01],
                         [0.0000e+00, -4.1421e-01],
                         [5.0000e-01, -7.0711e-01]],
                        [[-2.0711e-01, -2.5288e-07],
                         [-5.0000e-01, -8.5787e-02],
                         [-2.9289e-01, 7.0711e-01],
                         [5.0000e-01, 8.5786e-02],
                         [5.0000e-01, 8.5786e-02],
                         [-7.0711e-01, -2.9289e-01],
                         [-5.0000e-01, -8.5787e-02],
                         [-2.9289e-01, 2.0711e-01]],
                        [[7.0711e-01, 0.0000e+00],
                         [1.0000e+00, 0.0000e+00],
                         [2.9289e-01, 0.0000e+00],
                         [-1.0000e+00, 0.0000e+00],
                         [-1.0000e+00, 0.0000e+00],
                         [7.0711e-01, 0.0000e+00],
                         [1.0000e+00, 0.0000e+00],
                         [7.9289e-01, 0.0000e+00]]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_center_false():
    """
    Feature: Test spectrogram parameter: center.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_center_false")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, window=WindowType.HANN,
                            center=False, pad_mode=BorderType.REFLECT)(np.array(wav, dtype="float"))
    result = np.array([[[1.2100e+02, 3.1354e+02, 1.6900e+02, 2.5000e+01, 1.0843e+01,
                         1.2100e+02],
                        [3.3914e+01, 9.4728e+01, 4.5914e+01, 9.9142e+00, 4.5858e+00,
                         3.3914e+01],
                        [1.7157e-01, 1.5000e+00, 1.7157e-01, 1.7157e-01, 5.0000e-01,
                         1.7157e-01],
                        [2.5736e-01, 5.8579e-01, 2.5736e-01, 2.5736e-01, 5.8579e-01,
                         2.5736e-01],
                        [1.0000e+00, 8.5787e-02, 1.0000e+00, 1.0000e+00, 5.0000e-01,
                         1.0000e+00]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_normalized_true():
    """
    Feature: Test spectrogram parameter: normalized.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_normalized_true")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, window=WindowType.HANN,
                            center=False, normalized=True, pad_mode=BorderType.REFLECT)(np.array(wav, dtype="float"))
    result = np.array([[[4.0333e+01, 1.0451e+02, 5.6333e+01, 8.3333e+00, 3.6144e+00,
                         4.0333e+01],
                        [1.1305e+01, 3.1576e+01, 1.5305e+01, 3.3047e+00, 1.5286e+00,
                         1.1305e+01],
                        [5.7191e-02, 5.0000e-01, 5.7191e-02, 5.7191e-02, 1.6667e-01,
                         5.7191e-02],
                        [8.5786e-02, 1.9526e-01, 8.5786e-02, 8.5786e-02, 1.9526e-01,
                         8.5786e-02],
                        [3.3333e-01, 2.8596e-02, 3.3333e-01, 3.3333e-01, 1.6667e-01,
                         3.3333e-01]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_inputrank_3():
    """
    Feature: Test spectrogram parameter: input rank.
    Description: Test input rank.
    Expectation: Success.
    """
    logger.info("test_spectrogram_inputrank_3")
    wav = np.array([[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]],
                    [[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1]]])
    out = audio.Spectrogram(n_fft=8, window=WindowType.HANN, pad_mode=BorderType.REFLECT)(np.array(wav, dtype="float"))
    result = np.array([[[[2.8015e+01, 1.2100e+02, 3.1354e+02, 1.6900e+02, 3.3558e+01],
                         [3.2145e+00, 3.3914e+01, 9.4728e+01, 4.5914e+01, 6.7145e+00],
                         [1.0000e+00, 1.7157e-01, 1.5000e+00, 1.7157e-01, 7.5000e-01],
                         [4.2893e-02, 2.5736e-01, 5.8579e-01, 2.5736e-01, 1.2868e-01],
                         [5.0000e-01, 1.0000e+00, 8.5787e-02, 1.0000e+00, 6.2868e-01]]],
                       [[[2.8015e+01, 1.2100e+02, 3.1354e+02, 1.6900e+02, 3.3558e+01],
                         [3.2145e+00, 3.3914e+01, 9.4728e+01, 4.5914e+01, 6.7145e+00],
                         [1.0000e+00, 1.7157e-01, 1.5000e+00, 1.7157e-01, 7.5000e-01],
                         [4.2893e-02, 2.5736e-01, 5.8579e-01, 2.5736e-01, 1.2868e-01],
                         [5.0000e-01, 1.0000e+00, 8.5787e-02, 1.0000e+00, 6.2868e-01]]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_winlength_7():
    """
    Feature: Test spectrogram parameter: win_length.
    Description: Test parameter.
    Expectation: Success.
    """
    logger.info("test_spectrogram_winlength_7")
    wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
    out = audio.Spectrogram(n_fft=8, win_length=7, window=WindowType.HANN,)(np.array(wav, dtype="float"))
    result = np.array([[[2.0140e+01, 4.9000e+01, 1.5006e+02, 2.5284e+02, 1.5006e+02,
                         4.9000e+01, 4.5220e+00, 1.2250e+01, 7.6562e+01, 1.9600e+02,
                         2.7265e+02],
                        [5.0272e+00, 1.9488e+01, 5.5103e+01, 1.0153e+02, 5.5103e+01,
                         1.9488e+01, 2.0179e+00, 6.5089e+00, 2.9144e+01, 7.1406e+01,
                         1.0662e+02],
                        [4.2200e-01, 5.4020e-01, 1.0554e+00, 4.8321e+00, 1.0554e+00,
                         5.4020e-01, 9.6867e-01, 4.0345e-01, 7.8188e-01, 1.0872e+00,
                         3.3055e+00],
                        [1.2817e-01, 3.8618e-01, 2.1917e-01, 2.6102e-01, 2.1917e-01,
                         3.8618e-01, 4.3028e-01, 3.7738e-01, 2.0158e-01, 4.2135e-01,
                         5.7616e-02],
                        [3.7364e-01, 7.1574e-01, 8.1719e-01, 9.0949e-13, 8.1720e-01,
                         7.1573e-01, 2.7823e-01, 7.1573e-01, 8.1719e-01, 7.1574e-01,
                         3.7364e-01]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectrogram_param():
    """
    Feature: Test spectrogram invalid parameter.
    Description: Test some invalid parameters.
    Expectation: Success.
    """
    try:
        _ = audio.Spectrogram(n_fft=-1)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(n_fft=0)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(win_length=-1)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(win_length="s")
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.Spectrogram(hop_length=-1)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(hop_length=-100)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(win_length=300, n_fft=200)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)
    try:
        _ = audio.Spectrogram(pad=-1)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input pad is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.Spectrogram(power=-1)
    except ValueError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Input power is not within the required interval of [0, 16777216]." in str(error)
    try:
        _ = audio.Spectrogram(n_fft=False)
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument n_fft with value False is not of type (<class 'int'>,)" in str(error)
    try:
        _ = audio.Spectrogram(n_fft="s")
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument n_fft with value s is not of type [<class 'int'>], but got <class 'str'>." \
               in str(error)
    try:
        _ = audio.Spectrogram(window=False)
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument window with value False is not of type [<enum 'WindowType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.Spectrogram(pad_mode=False)
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument pad_mode with value False is not of type [<enum 'BorderType'>], but got <class 'bool'>." \
               in str(error)
    try:
        _ = audio.Spectrogram(onesided="s")
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument onesided with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.Spectrogram(center="s")
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument center with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.Spectrogram(normalized="s")
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument normalized with value s is not of type [<class 'bool'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.Spectrogram(normalized=1)
    except TypeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Argument normalized with value 1 is not of type [<class 'bool'>], but got <class 'int'>." in str(error)
    try:
        wav = np.array([[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]])
        _ = audio.Spectrogram(n_fft=100, center=False)(wav)
    except RuntimeError as error:
        logger.info("Got an exception in Spectrogram: {}".format(str(error)))
        assert "Unexpected error. Spectrogram: n_fft should be more than 0 and less than 30," \
               " but got n_fft: 100." in str(error)


if __name__ == "__main__":
    test_spectrogram_pipeline()
    test_spectrogram_eager()
    test_spectrogram_window_hamming_padmode_constant()
    test_spectrogram_nfft_10_window_bartlett_padmode_edge()
    test_spectrogram_onsided_false()
    test_spectrogram_power_0()
    test_spectrogram_center_false()
    test_spectrogram_normalized_true()
    test_spectrogram_inputrank_3()
    test_spectrogram_winlength_7()
    test_spectrogram_param()
