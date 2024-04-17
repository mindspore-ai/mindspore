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
import os
import copy
import numpy as np
from PIL import Image

import mindspore.common.dtype as mstype
import mindspore.dataset as ds
import mindspore.dataset.audio as audio
import mindspore.dataset.text as text
import mindspore.dataset.transforms as transforms
import mindspore.dataset.vision as vision
from mindspore.dataset.text import JiebaMode, NormalizeForm, SentencePieceModel, SPieceTokenizerOutType
from mindspore.dataset.transforms import Relational
from mindspore.dataset.vision import Border, Inter


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


def test_transforms_transform_input_consistency():
    """
    Feature: Transforms transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    data1 = np.array([1, 2, 3])
    data1_copy = copy.deepcopy(data1)
    data2 = [1, 2, 3]
    data2_copy = copy.deepcopy(data2)

    def check_result_any(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert (output != inputs_copy).any()

    def check_result(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert output != inputs_copy

    def check_result2(output, inputs, inputs_copy):
        assert inputs == inputs_copy
        assert (output != np.array(inputs_copy)).any()

    def check_result3(output, inputs, inputs_copy):
        assert inputs == inputs_copy
        assert output != inputs_copy

    out = transforms.Compose([transforms.Fill(10), transforms.Mask(Relational.EQ, 100)])(data1_copy)
    check_result_any(out, data1, data1_copy)

    pre_array = np.array([10, 20])
    append_array = np.array([100])
    out = transforms.Concatenate(axis=0, prepend=pre_array, append=append_array)(data1_copy)
    check_result(out, data1, data1_copy)

    out = transforms.Duplicate()(data2_copy)
    check_result3(out, data2, data2_copy)

    out = transforms.Fill(100)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = transforms.Mask(Relational.EQ, 2)(data2_copy)
    check_result2(out, data2, data2_copy)

    out = transforms.OneHot(num_classes=5, smoothing_rate=0)(data1_copy)
    check_result(out, data1, data1_copy)

    out = transforms.PadEnd(pad_shape=[4], pad_value=10)(data2_copy)
    check_result3(out, data2, data2_copy)

    data = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
    data_copy = copy.deepcopy(data)
    transform = [vision.HsvToRgb(is_hwc=True), vision.Crop((0, 0), 10), vision.ToTensor()]
    out = transforms.RandomApply(transform, prob=1.0)(data_copy)
    check_result(out, data, data_copy)

    out = transforms.RandomChoice([transforms.Fill(100)])(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = transforms.RandomOrder([transforms.Mask(Relational.EQ, 100)])(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = transforms.Slice(slice(1, 3))(data1_copy)
    check_result(out, data1, data1_copy)

    data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
    data_copy = copy.deepcopy(data)
    out = transforms.TypeCast(np.float16)(data_copy)
    check_result_any(out, data, data_copy)

    data = [[0, -1, -2, -1, 2], [2, -0, 2, 1, -3]]
    data_copy = copy.deepcopy(data)
    out = transforms.Unique()(data_copy)
    check_result3(out, data, data_copy)

    ds.config.set_seed(seed)


def test_vision_transform_input_consistency():
    """
    Feature: Vision transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    data1 = np.random.randint(0, 255, size=(100, 100, 3)).astype(np.uint8)
    data1_copy = copy.deepcopy(data1)
    data2 = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 2, 3))
    data2_copy = copy.deepcopy(data2)
    data3 = np.array([[0, 1, 2, 3, 4, 5], [0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((3, 4))
    data3_copy = copy.deepcopy(data3)
    data4 = np.fromfile("../data/dataset/apple.jpg", dtype=np.uint8)
    data4_copy = copy.deepcopy(data4)
    data5 = Image.open("../data/dataset/apple.jpg")
    data5_copy = copy.deepcopy(data5)
    data6 = np.array(Image.open("../data/dataset/apple.jpg"))
    data6_copy = copy.deepcopy(data6)

    def check_result_any(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert (output != inputs_copy).any()

    def check_result(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert output != inputs_copy

    def check_result2(output, inputs, inputs_copy):
        assert (np.array(inputs) == np.array(inputs_copy)).all()
        assert output != np.array(inputs_copy)

    data = np.random.randint(0, 256, (20, 20, 3)) / 255.0
    data = data.astype(np.float32)
    data_copy = copy.deepcopy(data)
    out = vision.AdjustBrightness(2.666)(data_copy)
    check_result_any(out, data, data_copy)

    out = vision.AdjustContrast(2.0)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.AdjustGamma(gamma=0.1, gain=1.0)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.AdjustHue(hue_factor=0.2)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.AdjustSaturation(saturation_factor=2.0)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.AdjustSharpness(sharpness_factor=0)(data3_copy)
    check_result_any(out, data3, data3_copy)

    out = vision.Affine(degrees=15, translate=[0.2, 0.2], scale=1.1,
                        shear=[1.0, 1.0], resample=Inter.BILINEAR)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.AutoAugment()(data1_copy)
    assert (data1 == data1_copy).all()
    assert out.shape == (100, 100, 3)

    out = vision.AutoContrast(cutoff=10.0, ignore=[10, 20])(data2_copy)
    check_result_any(out, data2, data2_copy)

    data = copy.deepcopy(data3_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.BoundingBoxAugment(transforms.Fill(100), 1.0)(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.CenterCrop(1)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.ConvertColor(vision.ConvertMode.COLOR_RGB2GRAY)(data2_copy)
    check_result(out, data2, data2_copy)

    out = vision.Crop((0, 0), 1)(data2_copy)
    check_result_any(out, data2, data2_copy)

    data = np.random.randint(0, 255, (3, 3, 10, 10)).astype(np.uint8)
    data_copy = copy.deepcopy(data)
    label = np.array([[0, 1], [1, 0], [1, 0]])
    out = vision.CutMixBatch(vision.ImageBatchFormat.NCHW, 1.0, 1.0)(data_copy, label)
    check_result(out, data, data_copy)

    out = vision.CutOut(20)(data1_copy)
    assert (data1 == data1_copy).all()
    assert out.shape == (100, 100, 3)

    out = vision.Decode()(data4_copy)
    check_result(out, data4, data4_copy)

    out = vision.Equalize()(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.Erase(0, 0, 2, 1)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.FiveCrop(2)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.GaussianBlur(3, 3)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.Grayscale(3)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.HorizontalFlip()(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.HsvToRgb(is_hwc=True)(data2_copy)
    check_result_any(out, data2, data2_copy)

    out = vision.HWC2CHW()(data2_copy)
    check_result(out, data2, data2_copy)

    out = vision.Invert()(data2_copy)
    check_result_any(out, data2, data2_copy)

    data = np.random.randn(10, 10, 3)
    data_copy = copy.deepcopy(data)
    transformation_matrix = np.random.randn(300, 300)
    mean_vector = np.random.randn(300,)
    out = vision.LinearTransformation(transformation_matrix=transformation_matrix, mean_vector=mean_vector)(data_copy)
    check_result_any(out, data, data_copy)

    label = np.array([[0, 1]])
    out = vision.MixUp(batch_size=2, alpha=0.2, is_single=False)(data1_copy, label)
    check_result(out, data1, data1_copy)

    data = np.random.randint(0, 255, (2, 10, 10, 3)).astype(np.uint8)
    data_copy = copy.deepcopy(data)
    label = np.array([[0, 1], [1, 0]])
    out = vision.MixUpBatch(1)(data_copy, label)
    check_result(out, data, data_copy)

    out = vision.Normalize(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0])(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.NormalizePad(mean=[121.0, 115.0, 100.0], std=[70.0, 68.0, 71.0], dtype="float32")(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.Pad([100, 100, 100, 100])(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.PadToSize([256, 256])(data1_copy)
    check_result(out, data1, data1_copy)

    start_points = [[0, 63], [63, 63], [63, 0], [0, 0]]
    end_points = [[0, 32], [32, 32], [32, 0], [0, 0]]
    out = vision.Perspective(start_points, end_points, Inter.BILINEAR)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.Posterize(4)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandAugment(interpolation=Inter.BILINEAR, fill_value=255)(data5_copy)
    check_result_any(out, np.array(data5), np.array(data5_copy))

    out = vision.RandomAdjustSharpness(2.0, 1.0)(data5_copy)
    check_result_any(out, np.array(data5), np.array(data5_copy))

    out = vision.RandomAffine(degrees=15, translate=(-0.1, 0.1, 0, 0),
                              scale=(0.9, 1.1), resample=Inter.NEAREST)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.RandomAutoContrast(cutoff=0.0, ignore=None, prob=1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandomColor((0.1, 1.9))(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.RandomColorAdjust(brightness=(0.5, 1), contrast=(0.4, 1), saturation=(0.3, 1))(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandomCrop(8, [10, 10, 10, 10], padding_mode=Border.EDGE)(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.RandomCropDecodeResize(size=(500, 520), scale=(0, 10.0), ratio=(0.5, 0.5),
                                        interpolation=Inter.BILINEAR, max_attempts=1)(data4_copy)
    check_result(out, data4, data4_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.RandomCropWithBBox([512, 512], [200, 200, 200, 200])(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.RandomEqualize(1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    data = np.random.randint(254, 255, size=(3, 100, 100)).astype(np.uint8)
    data_copy = copy.deepcopy(data)
    out = vision.RandomErasing(prob=1.0, max_attempts=1)(data_copy)
    check_result_any(out, data, data_copy)

    out = vision.RandomGrayscale(1.0)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.RandomHorizontalFlip(1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.RandomHorizontalFlipWithBBox(1)(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.RandomInvert(1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandomLighting(0.1)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.RandomPerspective(prob=1.0)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.RandomPosterize(1)(data5_copy)
    check_result_any(out, np.array(data5), np.array(data5_copy))

    out = vision.RandomResizedCrop(size=(50, 75), scale=(0.25, 0.5), interpolation=Inter.BILINEAR)(data1_copy)
    check_result(out, data1, data1_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.RandomResizedCropWithBBox((256, 512), (0.5, 0.5), (0.5, 0.5))(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.RandomResize(10)(data1_copy)
    check_result(out, data1, data1_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.RandomResizeWithBBox(100)(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.RandomRotation(degrees=90, resample=Inter.NEAREST, expand=True)(data1_copy)
    check_result(out, data1, data1_copy)

    policy = [[(vision.RandomRotation((90, 90)), 1),
               (vision.RandomColorAdjust(), 1)]]
    out = vision.RandomSelectSubpolicy(policy)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.RandomSharpness(degrees=(0, 0.6))(data5_copy)
    check_result(out, np.array(data5), np.array(data5_copy))

    out = vision.RandomSolarize(threshold=(1, 10))(data1_copy)
    check_result_any(out, data1, data1_copy)

    data = np.array([[0, 1, 2, 3, 4, 5]], dtype=np.uint8).reshape((2, 3))
    data_copy = copy.deepcopy(data)
    out = vision.RandomVerticalFlip(1.0)(data_copy)
    check_result_any(out, data, data_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.RandomVerticalFlipWithBBox(1)(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.Rescale(1.0 / 255.0, -1.0)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.Resize([5, 5], Inter.BICUBIC)(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.ResizedCrop(0, 0, 1, 1, (5, 5), Inter.BILINEAR)(data1_copy)
    check_result(out, data1, data1_copy)

    data = copy.deepcopy(data6_copy.astype(np.float32))
    func = lambda img, bboxes: (data, np.array([[0, 0, data.shape[1], data.shape[0]]]).astype(bboxes.dtype))
    func_data, func_bboxes = func(data, data)
    func_data_copy = copy.deepcopy(func_data)
    out = vision.ResizeWithBBox(100)(func_data_copy, func_bboxes)
    check_result(out, func_data, func_data_copy)

    out = vision.RgbToHsv(is_hwc=True)(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.Rotate(degrees=30.0, resample=Inter.NEAREST, expand=True)(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.SlicePatches(1, 2)(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.Solarize(threshold=(1, 10))(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.TenCrop(size=200)(data5_copy)
    check_result2(out, data5, data5_copy)

    out = vision.ToNumpy()(data5_copy)
    assert (np.array(data5_copy) == np.array(data5)).all()
    assert (out == np.array(data5_copy)).all()

    out = vision.ToPIL()(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.ToTensor()(data1_copy)
    check_result(out, data1, data1_copy)

    data = np.array([2.71606445312564e-03, 6.3476562564e-03]).astype(np.float64)
    data_copy = copy.deepcopy(data)
    out = vision.ToType(np.float32)(data_copy)
    check_result_any(out, data, data_copy)

    out = vision.TrivialAugmentWide()(data1_copy)
    check_result_any(out, data1, data1_copy)

    transforms_ua = [vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32]),
                     vision.RandomCrop(size=[200, 400], padding=[32, 32, 32, 32])]
    out = vision.UniformAugment(transforms_ua)(data6_copy)
    check_result(out, data6, data6_copy)

    out = vision.VerticalFlip()(data1_copy)
    check_result_any(out, data1, data1_copy)

    out = vision.encode_jpeg(data1_copy)
    check_result(out, data1, data1_copy)

    out = vision.encode_png(data1_copy)
    check_result(out, data1, data1_copy)

    data = np.array([1, 2, 3]).astype(np.uint8)
    data_copy = copy.deepcopy(data)
    filename = "./test_write_file.txt"
    vision.write_file(filename, data_copy)
    os.remove(filename)
    assert (data == data_copy).all()

    filename = "../test_write_jpeg.jpeg"
    vision.write_jpeg(filename, data1_copy)
    os.remove(filename)
    assert (data1 == data1_copy).all()

    filename = "../test_write_png.png"
    vision.write_png(filename, data1_copy)
    os.remove(filename)
    assert (data1 == data1_copy).all()

    ds.config.set_seed(seed)


def test_text_transform_input_consistency():
    """
    Feature: Text transform input consistency test
    Description: Check that operator execution does not affect the consistency of input data
    Expectation: Output is equal to the expected output
    """
    seed = ds.config.get_seed()
    ds.config.set_seed(12345)

    data1 = ["happy", "birthday", "to", "you"]
    data1_copy = copy.deepcopy(data1)
    data2 = 'Welcome     To   BeiJing!'
    data2_copy = copy.deepcopy(data2)
    data3 = "床前明月光"
    data3_copy = copy.deepcopy(data3)
    data4 = "123"
    data4_copy = copy.deepcopy(data4)

    def check_result(output, inputs, inputs_copy):
        assert inputs == inputs_copy
        assert output != inputs_copy

    def check_result2(output, inputs, inputs_copy):
        assert (inputs == inputs_copy).all()
        assert output != inputs_copy

    def check_result_len(output, inputs, inputs_copy):
        assert inputs == inputs_copy
        assert len(output) != len(inputs_copy)

    def check_result_any(output, inputs, inputs_copy):
        assert inputs == inputs_copy
        assert (output != inputs_copy).any()

    out = text.AddToken(token='TOKEN', begin=True)(data1_copy)
    check_result_len(out, data1, data1_copy)

    out = text.BasicTokenizer()(data2_copy)
    check_result_any(out, data2, data2_copy)

    vocab_bert = [
        "床", "前", "明", "月", "光", "疑", "是", "地", "上", "霜", "举", "头", "望", "低", "思", "故", "乡",
        "繁", "體", "字", "嘿", "哈", "大", "笑", "嘻",
        "i", "am", "mak", "make", "small", "mistake", "##s", "during", "work", "##ing", "hour",
        "+", "/", "-", "=", "12", "28", "40", "16", " ", "I",
        "[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"
    ]
    vocab = text.Vocab.from_list(vocab_bert)
    tokenizer_op = text.BertTokenizer(vocab=vocab)
    out = []
    for i in data3_copy:
        res = tokenizer_op(i)
        out.append(res)
    check_result(out, data3, data3_copy)

    out = text.CaseFold()(data2_copy)
    check_result(out, data2, data2_copy)

    out = text.FilterWikipediaXML()(data2_copy)
    check_result(out, data2, data2_copy)

    HMM_FILE = "../data/dataset/jiebadict/hmm_model.utf8"
    MP_FILE = "../data/dataset/jiebadict/jieba.dict.utf8"
    out = text.JiebaTokenizer(HMM_FILE, MP_FILE, mode=JiebaMode.MP)(data3_copy)
    check_result_any(out, data3, data3_copy)

    vocab = text.Vocab.from_list(["?", "##", "with", "the", "test", "符号"])
    data = "with"
    data_copy = copy.deepcopy(data)
    out = text.Lookup(vocab=vocab, unknown_token="test")(data_copy)
    check_result(out, data, data_copy)

    out = text.Ngram(3, separator=" ")(data1_copy)
    check_result_len(out, data1, data1_copy)

    data = ["ṩ", "ḍ̇", "q̇", "ﬁ", "2⁵", "ẛ"]
    data_copy = copy.deepcopy(data)
    out = text.NormalizeUTF8(NormalizeForm.NFKC)(data_copy)
    check_result_any(out, data, data_copy)

    def pytoken_op(input_data):
        return input_data.split()
    data = np.array('Hello world'.encode())
    data_copy = copy.deepcopy(data)
    out = text.PythonTokenizer(pytoken_op)(data_copy)
    check_result(out, data, data_copy)

    data = 'onetwoonetwoone'
    data_copy = copy.deepcopy(data)
    out = text.RegexReplace(pattern="one", replace="two", replace_all=True)(data_copy)
    check_result(out, data, data_copy)

    out = text.RegexTokenizer(delim_pattern="To", keep_delim_pattern="To", with_offsets=True)(data2_copy)
    check_result(out, data2, data2_copy)

    VOCAB_FILE = "../data/dataset/test_sentencepiece/vocab.txt"
    vocab = text.SentencePieceVocab.from_file([VOCAB_FILE], 100, 0.9995, SentencePieceModel.UNIGRAM, {})
    out = text.SentencePieceTokenizer(vocab, out_type=SPieceTokenizerOutType.STRING)(data4_copy)
    check_result_any(out, data4, data4_copy)

    out = text.SlidingWindow(2, 0)(data1_copy)
    check_result_len(out, data1, data1_copy)

    out = text.ToNumber(mstype.uint32)(data4_copy)
    check_result(out, data4, data4_copy)

    DATASET_ROOT_PATH = "../data/dataset/testVectors/"
    vectors = text.Vectors.from_file(DATASET_ROOT_PATH + "vectors.txt")
    out = text.ToVectors(vectors)(data1_copy)
    check_result(out, data1, data1_copy)

    out = text.Truncate(2)(data1_copy)
    check_result_len(out, data1, data1_copy)

    data = [["1", "2", "3"], ["4", "5"]]
    data_copy = copy.deepcopy(data)
    out = text.TruncateSequencePair(4)(*data_copy)
    check_result(out, data, data_copy)

    out = text.UnicodeCharTokenizer(with_offsets=True)(data2_copy)
    check_result(out, data2, data2_copy)

    unicode_script_tokenizer_op = text.UnicodeScriptTokenizer(keep_whitespace=True, with_offsets=False)
    out = []
    for i in data2_copy:
        out.append(unicode_script_tokenizer_op(i))
    assert data2 == data2_copy
    assert (np.array(out) != np.array(data2_copy)).any()

    out = text.WhitespaceTokenizer(with_offsets=True)(data2_copy)
    check_result(out, data2, data2_copy)

    vocab_list = ["book", "cholera", "era", "favor", "**ite", "my", "is", "love", "dur", "**ing", "the"]
    vocab = text.Vocab.from_list(vocab_list)
    out = text.WordpieceTokenizer(vocab=vocab, suffix_indicator="y", unknown_token='[UNK]')(data1_copy)
    check_result_any(out, data1, data1_copy)

    data = np.array(["This is a text file.", "Be happy every day.", "Good luck to everyone."])
    data_copy = copy.deepcopy(data)
    out = text.utils.to_bytes(data_copy)
    check_result2(out, data, data_copy)

    data = np.array(['4', '5', '6']).astype("S")
    data_copy = copy.deepcopy(data)
    out = text.to_str(data_copy, encoding='ascii')
    check_result2(out, data, data_copy)

    ds.config.set_seed(seed)
