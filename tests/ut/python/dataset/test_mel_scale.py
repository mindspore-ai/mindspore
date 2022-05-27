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
Testing MelScale op in DE
"""
import numpy as np
import pytest

import mindspore.dataset as ds
import mindspore.dataset.audio as c_audio
from mindspore import log as logger
from mindspore.dataset.audio.utils import MelType, NormType

CHANNEL = 1
FREQ = 20
TIME = 15
DEFAULT_N_MELS = 128


def gen(shape, dtype=np.float32):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=dtype),)


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


def test_mel_scale_pipeline():
    """
    Feature: MelScale
    Description: Test MelScale Cpp op in pipeline mode
    Expectation: Equal results from Mindspore and benchmark
    """
    in_data = np.array([[[[-0.34207549691200256, -2.0971477031707764, -0.9462487101554871],
                          [1.2536851167678833, -1.3225716352462769, -0.06942684203386307],
                          [-0.4859708547592163, -0.4990693926811218, 0.2322249710559845],
                          [-0.7589328289031982, -2.218672513961792, -0.8374152779579163]],
                         [[1.0313602685928345, -1.5596215724945068, 0.46823829412460327],
                          [0.14756731688976288, 0.35987502336502075, -1.3228634595870972],
                          [-0.7677955627441406, -0.059919968247413635, 0.7958201766014099],
                          [-0.6194286942481995, -0.5878928899765015, 0.3874965310096741]]]]).astype(np.float32)
    out_expect = np.array([[[-0.24386560916900635, -5.417530059814453, -1.4391992092132568],
                            [-0.08942853659391403, -0.7199308276176453, -0.18166661262512207]],
                           [[-0.0856514573097229, -1.6701887845993042, 0.25840121507644653],
                            [-0.12264516949653625, -0.1773705929517746, 0.07029043138027191]]]).astype(np.float32)
    dataset = ds.NumpySlicesDataset(in_data, column_names=["multi_dimensional_data"], shuffle=False)

    transforms = [c_audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=4)]
    dataset = dataset.map(operations=transforms, input_columns=["multi_dimensional_data"])

    for item in dataset.create_dict_iterator(num_epochs=1, output_numpy=True):
        out_put = item["multi_dimensional_data"]
        assert out_put.shape == (2, 2, 3)
        allclose_nparray(out_put, out_expect, 0.001, 0.001)


def test_mel_scale_pipeline_invalid_param():
    """
    Feature: MelScale
    Description: Test MelScale with invalid input parameters
    Expectation: Throw correct error and message
    """
    logger.info("test MelScale op with default values")
    generator = gen([CHANNEL, FREQ, TIME])
    data1 = ds.GeneratorDataset(source=generator, column_names=["multi_dimensional_data"])

    with pytest.raises(ValueError, match="MelScale: f_max should be greater than f_min."):
        transforms = [c_audio.MelScale(n_mels=128, sample_rate=16200, f_min=1000, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])
        for item in data1.create_dict_iterator(num_epochs=1, output_numpy=True):
            _ = item["multi_dimensional_data"]

    with pytest.raises(ValueError, match=r"Input n_mels is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.MelScale(n_mels=-1, sample_rate=16200, f_min=10, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    with pytest.raises(ValueError,
                       match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]."):
        transforms = [c_audio.MelScale(n_mels=128, sample_rate=0, f_min=10, f_max=1000)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    with pytest.raises(ValueError, match=r"Input f_max is not within the required interval of \(0, 16777216\]."):
        transforms = [c_audio.MelScale(n_mels=128, sample_rate=16200, f_min=10, f_max=-10)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    with pytest.raises(TypeError, match=r"Argument norm with value slaney is not of type \[<enum 'NormType'>\], " +
                       "but got <class 'str'>."):
        transforms = [c_audio.MelScale(n_mels=128, sample_rate=16200, f_min=10,
                                       f_max=1000, norm="slaney", mel_type=MelType.SLANEY)]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])

    with pytest.raises(TypeError, match=r"Argument mel_type with value SLANEY is not of type \[<enum 'MelType'>\], " +
                       "but got <class 'str'>."):
        transforms = [c_audio.MelScale(n_mels=128, sample_rate=16200, f_min=10, f_max=1000,
                                       norm=NormType.NONE, mel_type="SLANEY")]
        data1 = data1.map(operations=transforms, input_columns=["multi_dimensional_data"])


def test_mel_scale_eager():
    """
    Feature: MelScale
    Description: Test MelScale Cpp op with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.array([[[-0.7010437250137329, 1.1184569597244263, -1.4936821460723877],
                             [0.4603022038936615, -0.556514322757721, 0.8629537224769592]],
                            [[0.41759368777275085, 1.0594186782836914, -0.07423319667577744],
                             [0.47624683380126953, -0.33720797300338745, 2.0135815143585205]],
                            [[-0.6765501499176025, 0.8924005031585693, 1.0404413938522339],
                             [-0.5578446984291077, -0.349029004573822, 0.0370720773935318]]])
    spectrogram = spectrogram.astype(np.float32)
    out_ms = c_audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=2)(spectrogram)
    out_expect = np.array([[[-0.27036190032958984, 0.579207181930542, -0.6739760637283325],
                            [0.029620330780744553, -0.017264455556869507, 0.043247632682323456]],
                           [[0.7849390506744385, 0.706536054611206, 1.6048823595046997],
                            [0.10890152305364609, 0.01567467674612999, 0.33446595072746277]],
                           [[-1.0940029621124268, 0.5411258339881897, 1.000023603439331],
                            [-0.14039191603660583, 0.002245672047138214, 0.07748986035585403]]]).astype(np.float32)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)
    assert out_ms.shape == (3, 2, 3)

    spectrogram = np.array([[-0.7010437250137329, 1.1184569597244263, -1.4936821460723877],
                            [0.4603022038936615, -0.556514322757721, 0.8629537224769592]])
    spectrogram = spectrogram.astype(np.float32)
    out_ms = c_audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=2)(spectrogram)
    out_expect = np.array([[-0.27036190032958984, 0.579207181930542, -0.6739760637283325],
                           [0.029620330780744553, -0.017264455556869507, 0.043247632682323456]]).astype(np.float32)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)
    assert out_ms.shape == (2, 3)


if __name__ == "__main__":
    test_mel_scale_pipeline()
    test_mel_scale_pipeline_invalid_param()
    test_mel_scale_eager()
