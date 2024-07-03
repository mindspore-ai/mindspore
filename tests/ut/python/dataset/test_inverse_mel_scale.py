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
Testing InverseMelScale op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as c_audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import MelType, NormType

DATA_DIR = "../data/dataset/audiorecord/"


def get_ratio(mat):
    return mat.sum() / mat.size


def test_inverse_mel_scale_pipeline():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale cpp op in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    in_data = np.load(DATA_DIR + "inverse_mel_scale_8x40.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_20x40_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.InverseMelScale(n_stft=20, n_mels=8, sample_rate=8000,
                                          sgdargs={'sgd_lr': 0.05, 'sgd_momentum': 0.9})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-60
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2

    in_data = np.load(DATA_DIR + "inverse_mel_scale_4x80.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_40x80_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.InverseMelScale(n_stft=40, n_mels=4,
                                          sgdargs={'sgd_lr': 0.01, 'sgd_momentum': 0.9})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-60
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2

    in_data = np.load(DATA_DIR + "inverse_mel_scale_4x160.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_40x160_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [c_audio.InverseMelScale(n_stft=40, n_mels=4, f_min=10,
                                          sgdargs={'sgd_lr': 0.1, 'sgd_momentum': 0.8})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-60
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2


def test_inverse_mel_scale_pipeline_invalid_param():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("test InverseMelScale op with default values")
    in_data = np.load(DATA_DIR + "inverse_mel_scale_32x81.npy")[np.newaxis, :]
    data1 = ds.GeneratorDataset(in_data, column_names=["multi_dimensional_data"])
    # f_min and f_max
    with pytest.raises(ValueError,
                       match="MelScale: f_max should be greater than f_min."):
        transforms = [c_audio.InverseMelScale(n_mels=20, n_stft=128, sample_rate=16200, f_min=1000, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]
    # n_mel
    with pytest.raises(ValueError, match=r"Input n_mels is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.InverseMelScale(n_mels=-1, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # sample_rate
    with pytest.raises(ValueError,
                       match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=0, f_min=10, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # f_max
    with pytest.raises(ValueError, match=r"Input f_max is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # norm
    with pytest.raises(TypeError, match=r"Argument norm with value slaney is not of type \[<enum 'NormType'>\], " +
                       "but got <class 'str'>."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10,
                                              f_max=1000, norm="slaney", mel_type=MelType.SLANEY)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # mel_type
    with pytest.raises(TypeError, match=r"Argument mel_type with value SLANEY is not of type \[<enum 'MelType'>\], " +
                       "but got <class 'str'>."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                                              norm=NormType.NONE, mel_type="SLANEY")]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # max_iter
    with pytest.raises(ValueError, match=r"Input max_iter is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                                              norm=NormType.NONE, mel_type=MelType.SLANEY, max_iter=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # tolerance_loss
    with pytest.raises(ValueError,
                       match=r"Input tolerance_loss is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                                              norm=NormType.NONE, mel_type=MelType.SLANEY, tolerance_loss=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
    # tolerance_change
    with pytest.raises(ValueError,
                       match=r"Input tolerance_change is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                                              norm=NormType.NONE, mel_type=MelType.SLANEY, tolerance_change=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])


def test_inverse_mel_scale_eager():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale cpp op with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.load(DATA_DIR + 'inverse_mel_scale_32x81.npy')
    out_ms = c_audio.InverseMelScale(n_stft=80, n_mels=32)(spectrogram)
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_80x81_out.npy')

    epsilon = 1e-60
    relative_diff = np.abs((out_ms - out_expect) / (out_expect + epsilon))
    assert get_ratio(relative_diff < 1e-1) > 1e-2
    assert get_ratio(relative_diff < 1e-3) > 1e-3


if __name__ == "__main__":
    test_inverse_mel_scale_pipeline()
    test_inverse_mel_scale_pipeline_invalid_param()
    test_inverse_mel_scale_eager()
