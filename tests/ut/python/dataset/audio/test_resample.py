# Copyright 2021-2025 Huawei Technologies Co., Ltd
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
"""Test Resample."""

import numpy as np

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import ResampleMethod


def test_resample():
    """
    Feature: Resample
    Description: Test Resample with default arguments
    Expectation: the data is processed successfully
    """
    waveform_length = 30
    channel = 5
    waveform = np.random.random([channel, waveform_length])
    waveform = np.array([waveform])
    dataset = ds.NumpySlicesDataset(data=waveform, column_names=["audio"])
    transforms = [audio.Resample()]
    dataset = dataset.map(operations=transforms, input_columns=["audio"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["audio"]

    assert (out_put == waveform).all()


def test_resample_sinc_interpolation():
    """
    Feature: Resample
    Description: Test Resample with sinc_interpolation
    Expectation: the data is processed successfully
    """
    waveform = [[[9.1553e-05, 6.1035e-05, 6.1035e-05, 6.1035e-05, 1.2207e-04, 1.2207e-04,
                  9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 6.1035e-05,
                  1.2207e-04, 1.2207e-04, 1.2207e-04, 9.1553e-05, 9.1553e-05, 9.1553e-05,
                  6.1035e-05, 9.1553e-05]]]
    dataset_01 = ds.NumpySlicesDataset(data=waveform, column_names='audio', shuffle=False)
    transforms_01 = [audio.Resample(orig_freq=48000, new_freq=32000)]
    dataset_01 = dataset_01.map(operations=transforms_01, input_columns=["audio"])
    for item in dataset_01.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["audio"]

    expect_output = np.array([[7.217280358331832e-05, 6.147586845848083e-05, 6.933040265579067e-05,
                               0.0001266581382477944, 9.608767868756453e-05, 8.877784053662024e-05,
                               9.627620117997594e-05, 7.354420105365433e-05, 0.0001083820430709723,
                               0.0001270510740823957, 9.667916684509936e-05, 8.675770241390953e-05,
                               8.107692367547760e-05, 4.530917665365263e-05]], dtype=np.float64)
    print(output)
    assert np.mean(abs(output - expect_output)) < 0.0001


def test_resample_kaiser_window():
    """
    Feature: Resample
    Description: Test Resample with kaiser window
    Expectation: the data is processed successfully
    """
    waveform = [[[4.1653e-05, 2.10535e-05, 7.1335e-05, 4.1065e-05, 7.2507e-04,
                  3.4207e-04, 5.4553e-05, 3.1556e-05, 9.1553e-05, 9.1553e-05],
                 [9.1553e-05, 6.1035e-05, 1.2207e-04, 1.2457e-04, 1.2347e-04,
                  9.1673e-05, 9.1263e-05, 4.18753e-05, 8.1345e-05, 9.1675e-05]]]
    dataset_02 = ds.NumpySlicesDataset(data=waveform, column_names='audio', shuffle=False)
    transforms_02 = [audio.Resample(orig_freq=48000, new_freq=16000,
                                    resample_method=ResampleMethod.KAISER_WINDOW, lowpass_filter_width=7, rolloff=0.98)]
    dataset_02 = dataset_02.map(operations=transforms_02, input_columns=["audio"])
    for item in dataset_02.create_dict_iterator(num_epochs=1, output_numpy=True):
        output = item["audio"]
    expect_output = np.array([[-1.9667e-05, 2.7005e-04, 2.2237e-04, 1.4461e-05],
                              [5.4877e-05, 1.2215e-04, 8.5062e-05, 5.0822e-05]])
    assert np.mean(abs(output - expect_output)) < 0.0001


def test_resample_invalid_arguments():
    """
    Feature: Resample
    Description: Test Resample with wrong arguments
    Expectation: throw exception
    """
    try:
        audio.Resample(-1)
    except ValueError as e:
        assert "orig_freq is not within the required interval of " in str(e)
    try:
        audio.Resample(1, 0)
    except ValueError as e:
        assert "new_freq is not within the required interval of " in str(e)
    try:
        audio.Resample(resample_method="aaaa")
    except TypeError as e:
        assert "is not of type [<enum 'ResampleMethod'>]" in str(e)


def test_resample_eager():
    """
    Feature: Resample
    Description: Test Resample with eager
    Expectation: the data is processed successfully
    """
    waveform = np.array([[9.1553e-05, 6.1035e-05, 6.1035e-05, 6.1035e-05, 1.2207e-04, 1.2207e-04,
                          9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 9.1553e-05, 6.1035e-05,
                          1.2207e-04, 1.2207e-04, 1.2207e-04, 9.1553e-05, 9.1553e-05, 9.1553e-05,
                          6.1035e-05, 9.1553e-05]], dtype=np.float32)
    resample = audio.Resample(orig_freq=48000, new_freq=16000)
    output = resample(waveform)
    expect_output = np.array([[4.55755834991578e-05, 8.720954065211117e-05, 0.00010670421761460602,
                               8.14067243481986e-05, 0.0001065794640453532, 0.00010657223901944235,
                               7.18398878234438e-05]], dtype=np.float32)
    assert np.mean(abs(output - expect_output)) < 0.0001


if __name__ == '__main__':
    test_resample()
    test_resample_sinc_interpolation()
    test_resample_kaiser_window()
    test_resample_invalid_arguments()
    test_resample_eager()
