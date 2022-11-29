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
Testing LFCC Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import WindowType, BorderType, NormMode


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_lfcc_pipeline():
    """
    Feature: Test pipeline mode normal testcase: lfcc op
    Description: Input audio signal to test pipeline
    Expectation: Output is equal to the expected output
    """
    logger.info("test_lfcc_pipeline")

    wav = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.LFCC(sample_rate=16000, n_filter=128, n_lfcc=4, f_min=0.0, f_max=10000.0, dct_type=2,
                     norm=NormMode.ORTHO, log_lf=True,
                     speckwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "pad": 0,
                                 "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                 "pad_mode": BorderType.REFLECT, "onesided": True})
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["LFCC"])
    result = np.array([[[-137.9132, -137.0732, -137.2996, -140.0339],
                        [4.1616, 5.3870, 4.2134, 4.9916],
                        [-3.4581, -4.1653, -3.9544, -0.3347],
                        [2.0614, 2.7895, 2.7281, 0.7957]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["LFCC"], result, 0.0001, 0.0001)


def test_lfcc_eager():
    """
    Feature: Test eager mode normal testcase: lfcc op
    Description: Input audio signal to test eager
    Expectation: Output is equal to the expected output
    """
    logger.info("test_lfcc_eager")
    wav = np.array([[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]])
    out = audio.LFCC(sample_rate=16000, n_filter=128, n_lfcc=4, f_min=0.0, f_max=10000.0, dct_type=2,
                     norm=NormMode.ORTHO, log_lf=False,
                     speckwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "pad": 0,
                                 "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                 "pad_mode": BorderType.REFLECT, "onesided": True})(wav)
    result = np.array([[[[-5.5005e+02, -5.4640e+02, -5.4739e+02, -5.5840e+02],
                         [1.6892e+01, 2.2214e+01, 1.7117e+01, 2.0112e+01],
                         [-1.2942e+01, -1.6014e+01, -1.5098e+01, -3.6112e-01],
                         [8.1695e+00, 1.1332e+01, 1.1065e+01, 3.6742e+00]]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_lfcc_invalid_input():
    """
    Feature: LFCC op
    Description: Test operation with invalid input.
    Expectation: Throw exception as expected.
    """
    logger.info("test_lfcc_invalid_input")
    try:
        _ = audio.LFCC(sample_rate=-1)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.LFCC(sample_rate=1.1)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument sample_rate with value 1.1 is not of type [<class 'int'>]" in str(error)
    try:
        _ = audio.LFCC(n_filter=-1)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Input n_filter is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.LFCC(n_filter=1.1)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument n_filter with value 1.1 is not of type [<class 'int'>]" in str(error)
    try:
        _ = audio.LFCC(n_lfcc=-1)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Input n_lfcc is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.LFCC(n_lfcc=1.1)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument n_lfcc with value 1.1 is not of type [<class 'int'>]" in str(error)
    try:
        _ = audio.LFCC(log_lf=-1)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument log_lf with value -1 is not of type [<class 'bool'>]" in str(error)
    try:
        _ = audio.LFCC(norm="Karl Marx")
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument norm with value Karl Marx is not of type [<enum 'NormMode'>]" in str(error)
    try:
        _ = audio.LFCC(dct_type=-1)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "dct_type must be 2, but got : -1." in str(error)
    try:
        _ = audio.LFCC(f_min=10000)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "sample_rate // 2 should be greater than f_min when f_max is set to None" in str(error)
    try:
        _ = audio.LFCC(f_min=False)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument f_min with value False is not of type (<class 'int'>, <class 'float'>)" in str(error)
    try:
        _ = audio.LFCC(f_min=2, f_max=1)
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "f_max should be greater than or equal to f_min" in str(error)
    try:
        _ = audio.LFCC(f_max=False)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument f_max with value False is not of type (<class 'int'>, <class 'float'>)" in str(error)
    try:
        _ = audio.LFCC(speckwargs=False)
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument speckwargs with value False is not of type [<class 'dict'>]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": 0,
                                   "window": "WindowType.HANN", "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument window with value WindowType.HANN is not of type [<enum 'WindowType'>]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": 0,
                                   "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": 'BorderType.REFLECT', "onesided": True})
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument pad_mode with value BorderType.REFLECT is not of type [<enum 'BorderType'>]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": -1,
                                   "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Input pad is not within the required interval of [0, 2147483647]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": 1.1,
                                   "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument pad with value 1.1 is not of type [<class 'int'>]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": 0,
                                   "window": WindowType.HANN, "power": -1.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Input power is not within the required interval of [0, 16777216]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 400, "win_length": 16, "hop_length": 8, "pad": 0,
                                   "window": WindowType.HANN, "power": 2, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except TypeError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "Argument power with value 2 is not of type [<class 'float'>]" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 40, "win_length": 41, "hop_length": 8, "pad": 0,
                                   "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "win_length must be less than or equal to n_fft" in str(error)
    try:
        _ = audio.LFCC(speckwargs={"n_fft": 16, "win_length": 16, "hop_length": 8, "pad": 0,
                                   "window": WindowType.HANN, "power": 2.0, "normalized": False, "center": True,
                                   "pad_mode": BorderType.REFLECT, "onesided": True})
    except ValueError as error:
        logger.info("Got an exception in LFCC: {}".format(str(error)))
        assert "n_fft should be greater than or equal to n_lfcc" in str(error)


if __name__ == '__main__':
    test_lfcc_pipeline()
    test_lfcc_eager()
    test_lfcc_invalid_input()
