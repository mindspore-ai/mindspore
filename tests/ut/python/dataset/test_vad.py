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
Testing Vad op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio.transforms as c_audio
from mindspore import log as logger

DATA_DIR = "../data/dataset/audiorecord/"


def count_unequal_element(data_expected, data_me, rtol, atol):
    assert data_expected.shape == data_me.shape
    total_count = len(data_expected.flatten())
    error = np.abs(data_expected - data_me)
    greater = np.greater(error, atol + np.abs(data_expected) * rtol)
    loss_count = np.count_nonzero(greater)
    assert (loss_count / total_count) < rtol, \
        "\ndata_expected_std:{0}\ndata_me_error:{1}\nloss:{2}". \
        format(data_expected[greater], data_me[greater], error[greater])


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_vad_pipeline1():
    """
    Feature: Vad op
    Description: Test Vad op in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <1000>
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")[np.newaxis, :],
                                    column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=600)], input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(DATA_DIR + "single_channel_res.npy"), 0.001, 0.001)

    # <2, 1000>
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "double_channel.npy")
                                    [np.newaxis, :], column_names=["multi_dimensional_data"], shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=1600)], input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(DATA_DIR + "double_channel_res.npy"), 0.001, 0.001)

    # <1, 1000>
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.Vad(sample_rate=600)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(DATA_DIR + "single_channel_res.npy"), 0.001, 0.001)


def test_vad_pipeline2():
    """
    Feature: Vad op
    Description: Test Vad op in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <1, 1000> trigger level and time
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=700, trigger_level=14.0,
                                                  trigger_time=1.0)], input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_trigger_res.npy"), 0.001, 0.001)

    # <1, 1000> search time
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=750, trigger_level=14.0, trigger_time=1.0,
                                                  search_time=2.0)],
                          input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_search_res.npy"), 0.001, 0.001)

    # <1, 1000> allowed gap
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :],
                                    column_names=["multi_dimensional_data"], shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=750, trigger_level=14.0, trigger_time=1.0,
                                                  search_time=2.0, allowed_gap=0.125)],
                          input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_allowed_gap_res.npy"), 0.001, 0.001)

    # <1, 1000> boot time
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[
        c_audio.Vad(sample_rate=750,
                    trigger_level=14.0,
                    trigger_time=1.0,
                    search_time=2.0,
                    allowed_gap=0.125,
                    boot_time=0.7)
    ], input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_boot_time_res.npy"), 0.001, 0.001)


def test_vad_pipeline3():
    """
    Feature: Vad op
    Description: Test Vad op in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    # <1, 1000> noise
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=750,
                                                  trigger_level=14.0,
                                                  trigger_time=1.0,
                                                  search_time=2.0,
                                                  allowed_gap=0.125,
                                                  boot_time=0.7,
                                                  noise_up_time=0.5,
                                                  noise_down_time=0.1,
                                                  noise_reduction_amount=2.7)],
                          input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_noise_res.npy"), 0.001, 0.001)

    # <1, 1000> measure
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")
                                    [np.newaxis, np.newaxis, :], column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=800,
                                                  trigger_level=14.0,
                                                  trigger_time=1.0,
                                                  search_time=2.0,
                                                  allowed_gap=0.125,
                                                  boot_time=0.7,
                                                  noise_up_time=0.5,
                                                  noise_down_time=0.1,
                                                  noise_reduction_amount=2.7,
                                                  measure_freq=40,
                                                  measure_duration=0.05,
                                                  measure_smooth_time=1.0)], input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(
            DATA_DIR + "single_channel_measure_res.npy"), 0.001, 0.001)

    # <1, 1000> filter freq
    dataset = ds.NumpySlicesDataset(np.load(DATA_DIR + "single_channel.npy")[np.newaxis, np.newaxis, :],
                                    column_names=["multi_dimensional_data"],
                                    shuffle=False)
    dataset = dataset.map(operations=[c_audio.Vad(sample_rate=800,
                                                  trigger_level=14.0,
                                                  trigger_time=1.0,
                                                  search_time=2.0,
                                                  allowed_gap=0.125,
                                                  boot_time=0.7,
                                                  measure_freq=40,
                                                  measure_duration=0.05,
                                                  measure_smooth_time=1.0,
                                                  hp_filter_freq=20.0,
                                                  lp_filter_freq=3000.0,
                                                  hp_lifter_freq=75.0,
                                                  lp_lifter_freq=1000.0,
                                                  noise_up_time=0.5,
                                                  noise_down_time=0.1,
                                                  noise_reduction_amount=2.7)],
                          input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        allclose_nparray(item["multi_dimensional_data"], np.load(DATA_DIR + "single_channel_filter_res.npy"), 0.001,
                         0.001)


def test_vad_pipeline_invalid_param1():
    """
    Feature: Vad op
    Description: Test Vad with invalid input parameters
    Expectation: Throw ValueError or TypeError
    """
    logger.info("test InverseMelScale op with default values")
    in_data = np.load(DATA_DIR + "single_channel.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(ValueError,
                       match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.Vad(sample_rate=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input search_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, search_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input allowed_gap is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, allowed_gap=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input pre_trigger_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, pre_trigger_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input boot_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, boot_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_vad_pipeline_invalid_param2():
    """
    Feature: Vad op
    Description: Test Vad with invalid input parameters
    Expectation: Throw ValueError or TypeError
    """
    logger.info("test InverseMelScale op with default values")
    in_data = np.load(DATA_DIR + "single_channel.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(ValueError,
                       match=r"Input noise_up_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, noise_up_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input noise_down_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, noise_down_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input noise_up_time should be greater than noise_down_time, but got noise_up_time: 1 and"
                       + r" noise_down_time: 3."):
        transforms = [c_audio.Vad(sample_rate=1000, noise_up_time=1, noise_down_time=3)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input noise_reduction_amount is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, noise_reduction_amount=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_vad_pipeline_invalid_param3():
    """
    Feature: Vad op
    Description: Test Vad with invalid input parameters
    Expectation: Throw ValueError or TypeError
    """
    logger.info("test InverseMelScale op with default values")
    in_data = np.load(DATA_DIR + "single_channel.npy")[np.newaxis, :]
    data1 = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    with pytest.raises(ValueError,
                       match=r"Input measure_freq is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, measure_freq=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input measure_duration is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, measure_duration=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input measure_smooth_time is not within the required interval of \[0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, measure_smooth_time=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input hp_filter_freq is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, hp_filter_freq=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input lp_filter_freq is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, lp_filter_freq=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input hp_lifter_freq is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, hp_lifter_freq=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError,
                       match=r"Input lp_lifter_freq is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.Vad(sample_rate=1000, lp_lifter_freq=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]


def test_vad_eager():
    """
    Feature: Vad op
    Description: Test Vad op with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.load(DATA_DIR + "single_channel.npy")
    out_ms = c_audio.Vad(sample_rate=600)(spectrogram)
    out_expect = np.load(DATA_DIR + "single_channel_res.npy")
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)

    spectrogram = np.load(DATA_DIR + "double_channel.npy")
    out_ms = c_audio.Vad(sample_rate=1600)(spectrogram)
    out_expect = np.load(DATA_DIR + "double_channel_res.npy")
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)

    # benchmark op trigger warning
    spectrogram = np.load(DATA_DIR + "three_channel.npy")
    out_ms = c_audio.Vad(sample_rate=1600)(spectrogram)
    out_expect = np.load(DATA_DIR + "three_channel_res.npy")
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)


if __name__ == "__main__":
    test_vad_pipeline1()
    test_vad_pipeline2()
    test_vad_pipeline3()
    test_vad_pipeline_invalid_param1()
    test_vad_pipeline_invalid_param2()
    test_vad_pipeline_invalid_param3()
    test_vad_eager()
