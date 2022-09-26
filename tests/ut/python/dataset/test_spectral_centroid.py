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
Testing SpectralCentroid Python API
"""
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio
from mindspore import log as logger


def count_unequal_element(data_expected, data_me, rtol, atol):
    """ Precision calculation func """
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}".format(
        data_expected[greater], data_me[greater], error[greater])


def test_spectral_centroid_pipeline():
    """
    Feature: Mindspore pipeline mode normal testcase: spectral_centroid op.
    Description: Input audio signal to test pipeline.
    Expectation: Success.
    """
    logger.info("test_spectral_centroid_pipeline")

    wav = [[[1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]]]
    dataset = ds.NumpySlicesDataset(wav, column_names=["audio"], shuffle=False)
    out = audio.SpectralCentroid(sample_rate=44100, n_fft=8)
    dataset = dataset.map(operations=out, input_columns=["audio"], output_columns=["SpectralCentroid"])
    result = np.array([[[4436.1182, 3580.0718, 2902.4917, 3334.8962, 5199.8350, 6284.4814,
                         3580.0718, 2895.5659]]])
    for data1 in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        count_unequal_element(data1["SpectralCentroid"], result, 0.0001, 0.0001)


def test_spectral_centroid_eager():
    """
    Feature: Mindspore eager mode normal testcase: spectral_centroid op.
    Description: Input audio signal to test eager.
    Expectation: Success.
    """
    logger.info("test_spectral_centroid_eager")
    wav = np.array([[1.2, 1, 2, 2, 3, 3, 4, 4, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5.5, 6.5]])
    spectral_centroid_op = audio.SpectralCentroid(sample_rate=48000, n_fft=8)
    out = spectral_centroid_op(wav)
    result = np.array([[[5276.65022959, 3896.67543098, 3159.17400004, 3629.81957922,
                         5659.68456649, 6840.25126846, 3896.67543098, 3316.97434286]]])
    count_unequal_element(out, result, 0.0001, 0.0001)


def test_spectral_centroid_param():
    """
    Feature: Test spectral_centroid invalid parameter.
    Description: Test some invalid parameters.
    Expectation: Success.
    """
    try:
        _ = audio.SpectralCentroid(sample_rate=-1)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input sample_rate is not within the required interval of [0, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, n_fft=-1)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, n_fft=0)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input n_fft is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, win_length=-1)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input win_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, win_length="s")
    except TypeError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Argument win_length with value s is not of type [<class 'int'>], but got <class 'str'>." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, hop_length=-1)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, hop_length=-100)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input hop_length is not within the required interval of [1, 2147483647]." in str(error)
    try:
        _ = audio.SpectralCentroid(sample_rate=48000, win_length=300, n_fft=200)
    except ValueError as error:
        logger.info("Got an exception in SpectralCentroid: {}".format(str(error)))
        assert "Input win_length should be no more than n_fft, but got win_length: 300 and n_fft: 200." \
               in str(error)


if __name__ == "__main__":
    test_spectral_centroid_pipeline()
    test_spectral_centroid_eager()
    test_spectral_centroid_param()
