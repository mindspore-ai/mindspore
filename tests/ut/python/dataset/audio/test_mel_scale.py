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
"""Test MelScale."""

import numpy as np
import pytest

import mindspore.dataset as ds
from mindspore.dataset import audio
from mindspore.dataset.audio import MelType, NormType
from . import count_unequal_element

CHANNEL = 1
FREQ = 20
TIME = 15
DEFAULT_N_MELS = 128


def gen(shape, dtype=np.float32):
    np.random.seed(0)
    data = np.random.random(shape)
    yield (np.array(data, dtype=dtype),)


def allclose_nparray(data_expected, data_me, rtol, atol, equal_nan=True):
    if np.any(np.isnan(data_expected)):
        assert np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan)
    elif not np.allclose(data_me, data_expected, rtol, atol, equal_nan=equal_nan):
        count_unequal_element(data_expected, data_me, rtol, atol)


def test_mel_scale_pipeline():
    """
    Feature: MelScale
    Description: Test MelScale Cpp in pipeline mode
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

    transforms = [audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=4)]
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
    with pytest.raises(ValueError, match="MelScale: f_max should be greater than f_min."):
        audio.MelScale(n_mels=128, sample_rate=16200, f_min=1000, f_max=1000)

    with pytest.raises(ValueError, match=r"Input n_mels is not within the required interval of \[1, 2147483647\]."):
        audio.MelScale(n_mels=-1, sample_rate=16200, f_min=10, f_max=1000)

    with pytest.raises(ValueError,
                       match=r"Input sample_rate is not within the required interval of \[1, 2147483647\]."):
        audio.MelScale(n_mels=128, sample_rate=0, f_min=10, f_max=1000)

    with pytest.raises(ValueError, match=r"Input f_max is not within the required interval of \(0, 16777216\]."):
        audio.MelScale(n_mels=128, sample_rate=16200, f_min=10, f_max=-10)

    with pytest.raises(TypeError, match=r"Argument norm with value slaney is not of type \[<enum 'NormType'>\], " +
                                        "but got <class 'str'>."):
        audio.MelScale(n_mels=128, sample_rate=16200, f_min=10,
                       f_max=1000, norm="slaney", mel_type=MelType.SLANEY)

    with pytest.raises(TypeError, match=r"Argument mel_type with value SLANEY is not of type \[<enum 'MelType'>\], " +
                                        "but got <class 'str'>."):
        audio.MelScale(n_mels=128, sample_rate=16200, f_min=10, f_max=1000,
                       norm=NormType.NONE, mel_type="SLANEY")


def test_mel_scale_eager():
    """
    Feature: MelScale
    Description: Test MelScale Cpp with eager mode
    Expectation: Equal results from Mindspore and benchmark
    """
    spectrogram = np.array([[[-0.7010437250137329, 1.1184569597244263, -1.4936821460723877],
                             [0.4603022038936615, -0.556514322757721, 0.8629537224769592]],
                            [[0.41759368777275085, 1.0594186782836914, -0.07423319667577744],
                             [0.47624683380126953, -0.33720797300338745, 2.0135815143585205]],
                            [[-0.6765501499176025, 0.8924005031585693, 1.0404413938522339],
                             [-0.5578446984291077, -0.349029004573822, 0.0370720773935318]]])
    spectrogram = spectrogram.astype(np.float32)
    out_ms = audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=2)(spectrogram)
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
    out_ms = audio.MelScale(n_mels=2, sample_rate=10, f_min=-50, f_max=100, n_stft=2)(spectrogram)
    out_expect = np.array([[-0.27036190032958984, 0.579207181930542, -0.6739760637283325],
                           [0.029620330780744553, -0.017264455556869507, 0.043247632682323456]]).astype(np.float32)
    allclose_nparray(out_ms, out_expect, 0.001, 0.001)
    assert out_ms.shape == (2, 3)


def test_mel_scale_transform():
    """
    Feature: MelScale
    Description: Test MelScale with various valid input parameters and data types
    Expectation: The operation completes successfully
    """

    spectrum = np.array([[0.27318069338798523, 0.48482003808021545, -0.9954423308372498, -1.3815653324127197],
                         [0.18484283983707428, 0.986707866191864, -0.8953439593315125, 1.1496220827102661],
                         [-0.03155713528394699, -0.5174753665924072, -1.2002267837524414, -0.9929121136665344]])
    mel_scale = audio.MelScale(n_mels=3, n_stft=3)
    output = mel_scale(spectrum)
    assert np.shape(output) == (3, 4)

    # Test eager input dimension/shape
    spectrum = np.array([[[0.028305845335125923, -1.2127711772918701, -0.7961411476135254, -0.5237561464309692],
                          [0.7709200382232666, 0.9050686955451965, -0.7945995926856995, 0.738683819770813],
                          [0.7662771344184875, 0.3635537922382355, -0.6526238322257996, -2.159864902496338],
                          [0.1943817287683487, -1.437264323234558, 0.07929579168558121, -0.4990133047103882]],
                         [[0.9332525134086609, -0.9013996720314026, -0.43126359581947327, -0.28599128127098083],
                          [-0.6646514534950256, -0.4631001651287079, -0.2818048596382141, 1.1615350246429443],
                          [1.3551509380340576, -0.13417042791843414, -1.0491364002227783, -0.89068603515625],
                          [2.689964771270752, -0.7501158714294434, -0.36034297943115234, -1.4243621826171875]],
                         [[-0.43715158104896545, -0.8403787612915039, 1.284407377243042, 0.35216015577316284],
                          [-0.6144725680351257, -0.5544664263725281, -1.5388950109481812, 0.6705102920532227],
                          [-0.4711695611476898, -0.7128618955612183, -1.1746413707733154, -0.4072408676147461],
                          [-0.05665138363838196, -0.49570992588996887, 0.29964497685432434, 1.2298990488052368]]])
    mel_scale = audio.MelScale(n_mels=2, n_stft=4)
    output = mel_scale(spectrum)
    assert np.shape(output) == (3, 2, 4)

    # Test eager input dimension/shape
    spectrum = np.array([[[[-0.661716878414154, -0.9307411909103394, -1.4549999237060547],
                           [-0.5058236122131348, 0.6205351948738098, 0.45974650979042053],
                           [1.092147946357727, 0.7027606964111328, 0.03588017821311951]],
                          [[0.2836255133152008, -1.0368036031723022, -1.5129624605178833],
                           [1.2000963687896729, 0.40441256761550903, 0.24698682129383087],
                           [-1.0462393760681152, -1.968813180923462, -0.7837345004081726]],
                          [[0.3891336917877197, 0.7492761015892029, -2.166520595550537],
                           [0.5820790529251099, 1.0627620220184326, 0.8141032457351685],
                           [-0.5908588171005249, -0.5053110718727112, 0.5986936688423157]]],
                         [[[-1.0761715173721313, -0.23131141066551208, -0.03225666284561157],
                           [-0.12382299453020096, 1.3958712816238403, -0.08397931605577469],
                           [-0.2130870372056961, -0.7794660925865173, 1.4602042436599731]],
                          [[0.08928966522216797, 1.0202350616455078, -0.22602516412734985],
                           [0.47927194833755493, -0.5984132289886475, 1.098021149635315],
                           [-0.08548546582460403, -1.0120713710784912, 2.4553470611572266]],
                          [[0.446585476398468, 0.31302735209465027, -0.47839438915252686],
                           [-0.1200169026851654, -0.9213280081748962, 0.5382236242294312],
                           [0.6276569962501526, 1.7886888980865479, 1.3707671165466309]]]])
    mel_scale = audio.MelScale(n_mels=1, n_stft=3)
    output = mel_scale(spectrum)
    assert np.shape(output) == (2, 3, 1, 3)

    # Test eager data type
    spectrum = np.array([[-1.4439642429351807, 0.5226094722747803, 0.525325357913971, -0.34214285016059875],
                         [1.383328914642334, 0.4681275486946106, -1.2083654403686523, 0.07753748446702957],
                         [-1.0766335725784302, 0.43600115180015564, 0.1553124636411667, 0.5286868810653687]])
    spectrum = spectrum.astype(np.float16)
    mel_scale = audio.MelScale(n_mels=1, n_stft=3)
    output = mel_scale(spectrum)
    assert np.shape(output) == (1, 4)

    # Test eager data type
    spectrum = np.array([[-1.4439642429351807, 0.5226094722747803, 0.525325357913971, -0.34214285016059875],
                         [1.383328914642334, 0.4681275486946106, -1.2083654403686523, 0.07753748446702957],
                         [-1.0766335725784302, 0.43600115180015564, 0.1553124636411667, 0.5286868810653687]])
    spectrum = spectrum.astype(np.float32)
    mel_scale = audio.MelScale(n_mels=1, n_stft=3)
    output = mel_scale(spectrum)
    assert np.shape(output) == (1, 4)

    # Test eager data type
    spectrum = np.array([[-1.4439642429351807, 0.5226094722747803, 0.525325357913971, -0.34214285016059875],
                         [1.383328914642334, 0.4681275486946106, -1.2083654403686523, 0.07753748446702957],
                         [-1.0766335725784302, 0.43600115180015564, 0.1553124636411667, 0.5286868810653687]])
    spectrum = spectrum.astype(np.float64)
    mel_scale = audio.MelScale(n_mels=1, n_stft=3)
    output = mel_scale(spectrum)
    assert np.shape(output) == (1, 4)

    # Test eager data type
    spectrum = np.array([[0.8453619480133057, -1.4699571132659912],
                         [-1.6005735397338867, -0.08438355475664139]])
    spectrum = spectrum.tolist()
    mel_scale = audio.MelScale(n_mels=1)
    with pytest.raises(TypeError, match="Input should be NumPy audio, got <class 'list'>."):
        mel_scale(spectrum)

    # Test with invalid n_mels parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument n_mels with value value is not of type [<class 'int'>], but got <class 'str'>."
        audio.MelScale('value', 16000, 0, 100, None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid sample_rate parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument sample_rate with value 16000 is not of type [<class 'int'>], but got <class 'str'>."
        audio.MelScale(128, '16000', 0, 100, None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid f_min parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument f_min with value 0 is not of type [<class 'int'>, <class 'float'>], but got <class 's" \
                     "tr'>."
        audio.MelScale(128, 16000, '0', 100, None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid f_max parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument f_max with value 100 is not of type [<class 'int'>, <class 'float'>], but got <class" \
                     " 'str'>."
        audio.MelScale(128, 16000, 0, '100', None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid n_stft parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument n_stft with value None is not of type [<class 'int'>], but got <class 'str'>."
        audio.MelScale(128, 16000, 0, 100, 'None', NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid norm parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument norm with value NormType.NONE is not of type [<enum 'NormType'>], but got <class 'str'>."
        audio.MelScale(128, 16000, 0, 100, 201, 'NormType.NONE', MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with invalid mel_type parameter type (string type)
    with pytest.raises(TypeError) as err_info:
        error_info = "Argument mel_type with value MelType.HTK is not of type [<enum 'MelType'>], but got <class 's" \
                     "tr'>."
        audio.MelScale(128, 16000, 0, 100, 201, NormType.NONE, 'MelType.HTK')
    assert error_info in str(err_info.value)

    # Test with negative n_mels parameter
    with pytest.raises(ValueError) as err_info:
        error_info = "Input n_mels is not within the required interval of [1, 2147483647]."
        audio.MelScale(-10, 16000, 0, 100, None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with negative sample_rate parameter
    with pytest.raises(ValueError) as err_info:
        error_info = "Input sample_rate is not within the required interval of [1, 2147483647]."
        audio.MelScale(128, -16000, 0, 100, None, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)

    # Test with f_max less than f_min
    with pytest.raises(ValueError, match="MelScale: f_max should be greater than f_min."):
        audio.MelScale(128, 16000, 10, 5, None, NormType.NONE, MelType.HTK)

    # Test with sample_rate // 2 not greater than f_min when f_max is None
    with pytest.raises(ValueError, match="MelScale: sample_rate // 2 should be greater than "
                                         "f_min when f_max is set to None."):
        audio.MelScale(128, 20, 20, None, None, NormType.NONE, MelType.HTK)

    # Test with negative n_stft parameter
    with pytest.raises(ValueError) as err_info:
        error_info = "Input n_stft is not within the required interval of [1, 2147483647]."
        audio.MelScale(128, 16000, 0, 100, -10, NormType.NONE, MelType.HTK)
    assert error_info in str(err_info.value)


if __name__ == "__main__":
    test_mel_scale_pipeline()
    test_mel_scale_pipeline_invalid_param()
    test_mel_scale_eager()
    test_mel_scale_transform()
