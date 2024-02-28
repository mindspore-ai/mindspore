# Copyright 2024 Huawei Technologies Co., Ltd
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
This is the test for op input consistency
"""
import copy
import numpy as np

import mindspore.dataset as ds
import mindspore.dataset.audio as audio


def test_audio_transform_input_consistency():
    """
    Feature: audio transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    waveform1 = np.random.random([16])
    waveform1_copy = copy.deepcopy(waveform1)
    waveform2 = np.random.random([16, 2])
    waveform2_copy = copy.deepcopy(waveform2)
    waveform3 = np.random.random([400 // 2 + 1, 30])
    waveform3_copy = copy.deepcopy(waveform3)
    waveform3_complex = np.random.random([400 // 2 + 1, 30, 2])
    waveform3_complex_copy = copy.deepcopy(waveform3_complex)
    waveform4 = np.random.random([16, 3])
    waveform4_copy = copy.deepcopy(waveform4)
    waveform5 = np.random.random([16, 3, 2])
    waveform5_copy = copy.deepcopy(waveform5)
    waveform6 = np.random.random([4, 16])
    waveform6_copy = copy.deepcopy(waveform6)
    waveform7 = np.random.random([500])
    waveform7_copy = copy.deepcopy(waveform7)
    waveform8 = np.random.random([20, 20])
    waveform8_copy = copy.deepcopy(waveform8)
    waveform9 = np.random.random([1000])
    waveform9_copy = copy.deepcopy(waveform9)

    def check_result_all(output, input_waveform, waveform_copy):
        assert (input_waveform == waveform_copy).all()
        assert (input_waveform != output).all()

    def check_result_any(output, input_waveform, waveform_copy):
        assert (input_waveform == waveform_copy).all()
        assert (input_waveform != output).any()

    def check_result_shape(output, input_waveform, waveform_copy):
        assert (input_waveform == waveform_copy).all()
        assert input_waveform.shape != output.shape

    output = audio.AllpassBiquad(44100, 200.0)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.AmplitudeToDB(stype=audio.ScaleType.POWER)(waveform3)
    check_result_all(output, waveform3, waveform3_copy)

    output = audio.Angle()(waveform2)
    check_result_shape(output, waveform2, waveform2_copy)

    output = audio.BandBiquad(44100, 200.0)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.BandpassBiquad(44100, 200.0)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.BandrejectBiquad(44100, 200.0)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.BassBiquad(44100, 200.0)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.Biquad(0.01, 0.02, 0.13, 1, 0.12, 0.3)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.ComplexNorm()(waveform2)
    check_result_shape(output, waveform2, waveform2_copy)

    output = audio.ComputeDeltas(win_length=7, pad_mode=audio.BorderType.EDGE)(waveform3)
    check_result_all(output, waveform3, waveform3_copy)

    output = audio.Contrast()(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.DBToAmplitude(0.5, 0.5)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.DCShift(0.5, 0.02)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.DeemphBiquad(44100)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.DetectPitchFrequency(30, 0.1, 3, 5, 25)(waveform1)
    check_result_shape(output, waveform1, waveform1_copy)

    output = audio.Dither()(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.EqualizerBiquad(44100, 1500, 5.5, 0.7)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.Fade(fade_in_len=3, fade_out_len=2, fade_shape=audio.FadeShape.LINEAR)(waveform1)
    check_result_any(output, waveform1, waveform1_copy)

    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.3, 0.2, 0.1]
    output = audio.Filtfilt(a_coeffs, b_coeffs)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.Flanger(44100)(waveform6)
    check_result_any(output, waveform6, waveform6_copy)

    output = audio.FrequencyMasking(iid_masks=True, freq_mask_param=1)(waveform2)
    check_result_any(output, waveform2, waveform2_copy)

    output = audio.Gain(1.2)(waveform1)
    check_result_any(output, waveform1, waveform1_copy)

    output = audio.GriffinLim(n_fft=400)(waveform3)
    check_result_shape(output, waveform3, waveform3_copy)

    output = audio.HighpassBiquad(44100, 1500, 0.7)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.InverseMelScale(20, 3, 16000, 0, 8000, 10)(waveform5)
    check_result_shape(output, waveform5, waveform5_copy)

    output = audio.InverseSpectrogram(1, 400, 400, 200)(waveform3_complex)
    check_result_all(output, waveform3_complex, waveform3_complex_copy)

    output = audio.LFCC()(waveform6)
    check_result_shape(output, waveform6, waveform6_copy)

    a_coeffs = [0.1, 0.2, 0.3]
    b_coeffs = [0.3, 0.2, 0.1]
    output = audio.LFilter(a_coeffs, b_coeffs)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.LowpassBiquad(4000, 1500, 0.7)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.Magphase()(waveform2)
    assert (waveform2 == waveform2_copy).all()
    assert isinstance(waveform2.shape, tuple)

    output = audio.MaskAlongAxis(0, 10, 0.5, 1)(waveform8)
    check_result_any(output, waveform8, waveform8_copy)

    output = audio.MaskAlongAxisIID(5, 0.5, 2)(waveform8)
    check_result_any(output, waveform8, waveform8_copy)

    output = audio.MelScale(6, 15, 0.7, None, 16)(waveform4)
    check_result_shape(output, waveform4, waveform4_copy)

    output = audio.MelSpectrogram(sample_rate=16000, n_fft=16, win_length=16, hop_length=8, f_min=0.0, \
                                  f_max=5000.0, pad=0, n_mels=3, window=audio.WindowType.HANN, power=2.0, \
                                  normalized=False, center=True, pad_mode=audio.BorderType.REFLECT, \
                                  onesided=True, norm=audio.NormType.SLANEY, mel_scale=audio.MelType.HTK)(waveform1)
    check_result_shape(output, waveform1, waveform1_copy)

    output = audio.MFCC(4000, 128, 2)(waveform7)
    check_result_shape(output, waveform7, waveform7_copy)

    output = audio.MuLawDecoding()(waveform6)
    check_result_all(output, waveform6, waveform6_copy)

    output = audio.MuLawEncoding()(waveform6)
    check_result_all(output, waveform6, waveform6_copy)

    output = audio.Overdrive()(waveform1)
    check_result_any(output, waveform1, waveform1_copy)

    output = audio.Phaser(44100)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    phase_advance = np.random.random([16, 1])
    output = audio.PhaseVocoder(rate=2, phase_advance=phase_advance)(waveform5)
    check_result_shape(output, waveform5, waveform5_copy)

    output = audio.PitchShift(sample_rate=16000, n_steps=4)(waveform6)
    check_result_all(output, waveform6, waveform6_copy)

    output = audio.Resample(orig_freq=48000, new_freq=16000,
                            resample_method=audio.ResampleMethod.SINC_INTERPOLATION,
                            lowpass_filter_width=6, rolloff=0.99, beta=None)(waveform6)
    check_result_shape(output, waveform6, waveform6_copy)

    output = audio.RiaaBiquad(44100)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    output = audio.SlidingWindowCmn()(waveform4)
    check_result_all(output, waveform4, waveform4_copy)

    output = audio.SpectralCentroid(44100)(waveform6)
    check_result_all(output, waveform6, waveform6_copy)

    output = audio.Spectrogram()(waveform6)
    check_result_shape(output, waveform6, waveform6_copy)

    output = audio.TimeMasking(time_mask_param=1)(waveform2)
    check_result_any(output, waveform2, waveform2_copy)

    output = audio.TimeStretch(5, 201, 2.0)(waveform5)
    check_result_shape(output, waveform5, waveform5_copy)

    output = audio.TrebleBiquad(44100, 200.0)(waveform1)
    check_result_any(output, waveform1, waveform1_copy)

    output = audio.Vad(sample_rate=600)(waveform9)
    check_result_shape(output, waveform9, waveform9_copy)

    output = audio.Vol(gain=10, gain_type=audio.GainType.DB)(waveform1)
    check_result_all(output, waveform1, waveform1_copy)

    ds.config.set_seed(seed)
