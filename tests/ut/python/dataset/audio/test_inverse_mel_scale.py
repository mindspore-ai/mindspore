# Copyright 2022-2025 Huawei Technologies Co., Ltd
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
"""Test InverseMelScale."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import MelType, NormType

DATA_DIR = "../data/dataset/audiorecord/"


def get_ratio(mat):
    return mat.sum() / mat.size


def test_inverse_mel_scale_pipeline():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale cpp in pipeline
    Expectation: Equal results from Mindspore and benchmark
    """
    in_data = np.load(DATA_DIR + "inverse_mel_scale_8x40.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_20x40_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.InverseMelScale(n_stft=20, n_mels=8, sample_rate=8000,
                                        sgdargs={'sgd_lr': 0.05, 'sgd_momentum': 0.9})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-6
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2

    in_data = np.load(DATA_DIR + "inverse_mel_scale_4x80.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_40x80_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.InverseMelScale(n_stft=40, n_mels=4,
                                        sgdargs={'sgd_lr': 0.01, 'sgd_momentum': 0.9})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-6
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2

    in_data = np.load(DATA_DIR + "inverse_mel_scale_4x160.npy")[np.newaxis, :]
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_40x160_out.npy')[np.newaxis, :]
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)
    transforms = [audio.InverseMelScale(n_stft=40, n_mels=4, f_min=10,
                                        sgdargs={'sgd_lr': 0.1, 'sgd_momentum': 0.8})]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])
    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_data = item["multi_dimensional_data"]
        epsilon = 1e-6
        relative_diff = np.abs((out_data - out_expect) / (out_expect + epsilon))
        assert get_ratio(relative_diff < 1e-1) > 1e-2


def test_inverse_mel_scale_pipeline_invalid_param():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale with invalid input parameters
    Expectation: Throw correct error and message
    """
    # f_min and f_max
    with pytest.raises(ValueError,
                       match="MelScale: f_max should be greater than f_min."):
        audio.InverseMelScale(n_mels=20, n_stft=128, sample_rate=16200, f_min=1000, f_max=1000)

    # n_mel
    with pytest.raises(ValueError, match=r"Input n_mels is not within the required interval of \[1, 2147483647\]."):
        audio.InverseMelScale(n_mels=-1, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000)

    # sample_rate
    with pytest.raises(ValueError,
                       match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=0, f_min=10, f_max=1000)

    # f_max
    with pytest.raises(ValueError, match=r"Input f_max is not within the required interval of \(0, 16777216\]."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=-10)

    # norm
    with pytest.raises(TypeError, match=r"Argument norm with value slaney is not of type \[<enum 'NormType'>\], " +
                       "but got <class 'str'>."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10,
                              f_max=1000, norm="slaney", mel_type=MelType.SLANEY)

    # mel_type
    with pytest.raises(TypeError, match=r"Argument mel_type with value SLANEY is not of type \[<enum 'MelType'>\], " +
                       "but got <class 'str'>."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                              norm=NormType.NONE, mel_type="SLANEY")

    # max_iter
    with pytest.raises(ValueError, match=r"Input max_iter is not within the required interval of \[1, 2147483647\]."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                              norm=NormType.NONE, mel_type=MelType.SLANEY, max_iter=-10)

    # tolerance_loss
    with pytest.raises(ValueError,
                       match=r"Input tolerance_loss is not within the required interval of \(0, 16777216\]."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                              norm=NormType.NONE, mel_type=MelType.SLANEY, tolerance_loss=-10)

    # tolerance_change
    with pytest.raises(ValueError,
                       match=r"Input tolerance_change is not within the required interval of \(0, 16777216\]."):
        audio.InverseMelScale(n_mels=128, n_stft=2000, sample_rate=16200, f_min=10, f_max=1000,
                              norm=NormType.NONE, mel_type=MelType.SLANEY, tolerance_change=-10)


def test_inverse_mel_scale_eager():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale cpp with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.load(DATA_DIR + 'inverse_mel_scale_32x81.npy')
    out_ms = audio.InverseMelScale(n_stft=80, n_mels=32)(spectrogram)
    out_expect = np.load(DATA_DIR + 'inverse_mel_scale_80x81_out.npy')

    epsilon = 1e-6
    relative_diff = np.abs((out_ms - out_expect) / (out_expect + epsilon))
    assert get_ratio(relative_diff < 1e-1) > 1e-2
    assert get_ratio(relative_diff < 1e-3) > 1e-3


def test_inverse_mel_scale_float64():
    """
    Feature: InverseMelScale
    Description: Test InverseMelScale on waveform in type of float64
    Expectation: The result does not contain nan
    """
    waveform = np.random.random([8, 3, 2]).astype(np.float64)
    result = audio.InverseMelScale(20, 3, 16000, 0, 8000, 10)(waveform)
    assert not np.any(np.isnan(result))


if __name__ == "__main__":
    test_inverse_mel_scale_pipeline()
    test_inverse_mel_scale_pipeline_invalid_param()
    test_inverse_mel_scale_eager()
    test_inverse_mel_scale_float64()
