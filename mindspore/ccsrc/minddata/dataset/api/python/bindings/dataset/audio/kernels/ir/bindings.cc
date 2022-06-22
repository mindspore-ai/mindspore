/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pybind11/pybind11.h"

#include "minddata/dataset/api/python/pybind_conversion.h"
#include "minddata/dataset/api/python/pybind_register.h"
#include "minddata/dataset/include/dataset/transforms.h"

#include "minddata/dataset/audio/ir/kernels/allpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/amplitude_to_db_ir.h"
#include "minddata/dataset/audio/ir/kernels/angle_ir.h"
#include "minddata/dataset/audio/ir/kernels/band_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bandreject_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/bass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/complex_norm_ir.h"
#include "minddata/dataset/audio/ir/kernels/compute_deltas_ir.h"
#include "minddata/dataset/audio/ir/kernels/contrast_ir.h"
#include "minddata/dataset/audio/ir/kernels/db_to_amplitude_ir.h"
#include "minddata/dataset/audio/ir/kernels/dc_shift_ir.h"
#include "minddata/dataset/audio/ir/kernels/deemph_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/detect_pitch_frequency_ir.h"
#include "minddata/dataset/audio/ir/kernels/dither_ir.h"
#include "minddata/dataset/audio/ir/kernels/equalizer_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/fade_ir.h"
#include "minddata/dataset/audio/ir/kernels/filtfilt_ir.h"
#include "minddata/dataset/audio/ir/kernels/flanger_ir.h"
#include "minddata/dataset/audio/ir/kernels/frequency_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/gain_ir.h"
#include "minddata/dataset/audio/ir/kernels/griffin_lim_ir.h"
#include "minddata/dataset/audio/ir/kernels/highpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/inverse_mel_scale_ir.h"
#include "minddata/dataset/audio/ir/kernels/lfilter_ir.h"
#include "minddata/dataset/audio/ir/kernels/lowpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/magphase_ir.h"
#include "minddata/dataset/audio/ir/kernels/mask_along_axis_iid_ir.h"
#include "minddata/dataset/audio/ir/kernels/mask_along_axis_ir.h"
#include "minddata/dataset/audio/ir/kernels/mel_scale_ir.h"
#include "minddata/dataset/audio/ir/kernels/mu_law_decoding_ir.h"
#include "minddata/dataset/audio/ir/kernels/mu_law_encoding_ir.h"
#include "minddata/dataset/audio/ir/kernels/overdrive_ir.h"
#include "minddata/dataset/audio/ir/kernels/phase_vocoder_ir.h"
#include "minddata/dataset/audio/ir/kernels/phaser_ir.h"
#include "minddata/dataset/audio/ir/kernels/resample_ir.h"
#include "minddata/dataset/audio/ir/kernels/riaa_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/sliding_window_cmn_ir.h"
#include "minddata/dataset/audio/ir/kernels/spectral_centroid_ir.h"
#include "minddata/dataset/audio/ir/kernels/spectrogram_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_stretch_ir.h"
#include "minddata/dataset/audio/ir/kernels/treble_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/vad_ir.h"
#include "minddata/dataset/audio/ir/kernels/vol_ir.h"

namespace mindspore {
namespace dataset {
PYBIND_REGISTER(
  AllpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::AllpassBiquadOperation, TensorOperation, std::shared_ptr<audio::AllpassBiquadOperation>>(
      *m, "AllpassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q) {
        auto allpass_biquad = std::make_shared<audio::AllpassBiquadOperation>(sample_rate, central_freq, Q);
        THROW_IF_ERROR(allpass_biquad->ValidateParams());
        return allpass_biquad;
      }));
  }));

PYBIND_REGISTER(
  AmplitudeToDBOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::AmplitudeToDBOperation, TensorOperation, std::shared_ptr<audio::AmplitudeToDBOperation>>(
      *m, "AmplitudeToDBOperation")
      .def(py::init([](ScaleType stype, float ref_value, float amin, float top_db) {
        auto amplitude_to_db = std::make_shared<audio::AmplitudeToDBOperation>(stype, ref_value, amin, top_db);
        THROW_IF_ERROR(amplitude_to_db->ValidateParams());
        return amplitude_to_db;
      }));
  }));

PYBIND_REGISTER(ScaleType, 0, ([](const py::module *m) {
                  (void)py::enum_<ScaleType>(*m, "ScaleType", py::arithmetic())
                    .value("DE_SCALE_TYPE_MAGNITUDE", ScaleType::kMagnitude)
                    .value("DE_SCALE_TYPE_POWER", ScaleType::kPower)
                    .export_values();
                }));

PYBIND_REGISTER(AngleOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::AngleOperation, TensorOperation, std::shared_ptr<audio::AngleOperation>>(
                    *m, "AngleOperation")
                    .def(py::init([]() {
                      auto angle = std::make_shared<audio::AngleOperation>();
                      THROW_IF_ERROR(angle->ValidateParams());
                      return angle;
                    }));
                }));

PYBIND_REGISTER(
  BandBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BandBiquadOperation, TensorOperation, std::shared_ptr<audio::BandBiquadOperation>>(
      *m, "BandBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q, bool noise) {
        auto band_biquad = std::make_shared<audio::BandBiquadOperation>(sample_rate, central_freq, Q, noise);
        THROW_IF_ERROR(band_biquad->ValidateParams());
        return band_biquad;
      }));
  }));

PYBIND_REGISTER(
  BandpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BandpassBiquadOperation, TensorOperation, std::shared_ptr<audio::BandpassBiquadOperation>>(
      *m, "BandpassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain) {
        auto bandpass_biquad =
          std::make_shared<audio::BandpassBiquadOperation>(sample_rate, central_freq, Q, const_skirt_gain);
        THROW_IF_ERROR(bandpass_biquad->ValidateParams());
        return bandpass_biquad;
      }));
  }));

PYBIND_REGISTER(BandrejectBiquadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::BandrejectBiquadOperation, TensorOperation,
                                   std::shared_ptr<audio::BandrejectBiquadOperation>>(*m, "BandrejectBiquadOperation")
                    .def(py::init([](int32_t sample_rate, float central_freq, float Q) {
                      auto bandreject_biquad =
                        std::make_shared<audio::BandrejectBiquadOperation>(sample_rate, central_freq, Q);
                      THROW_IF_ERROR(bandreject_biquad->ValidateParams());
                      return bandreject_biquad;
                    }));
                }));

PYBIND_REGISTER(
  BassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::BassBiquadOperation, TensorOperation, std::shared_ptr<audio::BassBiquadOperation>>(
      *m, "BassBiquadOperation")
      .def(py::init([](int32_t sample_rate, float gain, float central_freq, float Q) {
        auto bass_biquad = std::make_shared<audio::BassBiquadOperation>(sample_rate, gain, central_freq, Q);
        THROW_IF_ERROR(bass_biquad->ValidateParams());
        return bass_biquad;
      }));
  }));

PYBIND_REGISTER(BiquadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::BiquadOperation, TensorOperation, std::shared_ptr<audio::BiquadOperation>>(
                    *m, "BiquadOperation")
                    .def(py::init([](float b0, float b1, float b2, float a0, float a1, float a2) {
                      auto biquad = std::make_shared<audio::BiquadOperation>(b0, b1, b2, a0, a1, a2);
                      THROW_IF_ERROR(biquad->ValidateParams());
                      return biquad;
                    }));
                }));

PYBIND_REGISTER(
  ComplexNormOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::ComplexNormOperation, TensorOperation, std::shared_ptr<audio::ComplexNormOperation>>(
      *m, "ComplexNormOperation")
      .def(py::init([](float power) {
        auto complex_norm = std::make_shared<audio::ComplexNormOperation>(power);
        THROW_IF_ERROR(complex_norm->ValidateParams());
        return complex_norm;
      }));
  }));

PYBIND_REGISTER(
  ComputeDeltasOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::ComputeDeltasOperation, TensorOperation, std::shared_ptr<audio::ComputeDeltasOperation>>(
      *m, "ComputeDeltasOperation")
      .def(py::init([](int32_t win_length, BorderType pad_mode) {
        auto compute_deltas = std::make_shared<audio::ComputeDeltasOperation>(win_length, pad_mode);
        THROW_IF_ERROR(compute_deltas->ValidateParams());
        return compute_deltas;
      }));
  }));

PYBIND_REGISTER(ContrastOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::ContrastOperation, TensorOperation, std::shared_ptr<audio::ContrastOperation>>(
                      *m, "ContrastOperation")
                      .def(py::init([](float enhancement_amount) {
                        auto contrast = std::make_shared<audio::ContrastOperation>(enhancement_amount);
                        THROW_IF_ERROR(contrast->ValidateParams());
                        return contrast;
                      }));
                }));

PYBIND_REGISTER(
  DBToAmplitudeOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::DBToAmplitudeOperation, TensorOperation, std::shared_ptr<audio::DBToAmplitudeOperation>>(
      *m, "DBToAmplitudeOperation")
      .def(py::init([](float ref, float power) {
        auto db_to_amplitude = std::make_shared<audio::DBToAmplitudeOperation>(ref, power);
        THROW_IF_ERROR(db_to_amplitude->ValidateParams());
        return db_to_amplitude;
      }));
  }));

PYBIND_REGISTER(DCShiftOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::DCShiftOperation, TensorOperation, std::shared_ptr<audio::DCShiftOperation>>(
                    *m, "DCShiftOperation")
                    .def(py::init([](float shift, float limiter_gain) {
                      auto dc_shift = std::make_shared<audio::DCShiftOperation>(shift, limiter_gain);
                      THROW_IF_ERROR(dc_shift->ValidateParams());
                      return dc_shift;
                    }));
                }));

PYBIND_REGISTER(
  DeemphBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::DeemphBiquadOperation, TensorOperation, std::shared_ptr<audio::DeemphBiquadOperation>>(
      *m, "DeemphBiquadOperation")
      .def(py::init([](int32_t sample_rate) {
        auto deemph_biquad = std::make_shared<audio::DeemphBiquadOperation>(sample_rate);
        THROW_IF_ERROR(deemph_biquad->ValidateParams());
        return deemph_biquad;
      }));
  }));

PYBIND_REGISTER(DetectPitchFrequencyOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::DetectPitchFrequencyOperation, TensorOperation,
                                   std::shared_ptr<audio::DetectPitchFrequencyOperation>>(
                    *m, "DetectPitchFrequencyOperation")
                    .def(py::init([](int32_t sample_rate, float frame_time, int32_t win_length, int32_t freq_low,
                                     int32_t freq_high) {
                      auto detect_pitch_frequency = std::make_shared<audio::DetectPitchFrequencyOperation>(
                        sample_rate, frame_time, win_length, freq_low, freq_high);
                      THROW_IF_ERROR(detect_pitch_frequency->ValidateParams());
                      return detect_pitch_frequency;
                    }));
                }));

PYBIND_REGISTER(DensityFunction, 0, ([](const py::module *m) {
                  (void)py::enum_<DensityFunction>(*m, "DensityFunction", py::arithmetic())
                    .value("DE_DENSITY_FUNCTION_TPDF", DensityFunction::kTPDF)
                    .value("DE_DENSITY_FUNCTION_RPDF", DensityFunction::kRPDF)
                    .value("DE_DENSITY_FUNCTION_GPDF", DensityFunction::kGPDF)
                    .export_values();
                }));

PYBIND_REGISTER(DitherOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::DitherOperation, TensorOperation, std::shared_ptr<audio::DitherOperation>>(
                    *m, "DitherOperation")
                    .def(py::init([](DensityFunction density_function, bool noise_shaping) {
                      auto dither = std::make_shared<audio::DitherOperation>(density_function, noise_shaping);
                      THROW_IF_ERROR(dither->ValidateParams());
                      return dither;
                    }));
                }));

PYBIND_REGISTER(EqualizerBiquadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::EqualizerBiquadOperation, TensorOperation,
                                   std::shared_ptr<audio::EqualizerBiquadOperation>>(*m, "EqualizerBiquadOperation")
                    .def(py::init([](int sample_rate, float center_freq, float gain, float Q) {
                      auto equalizer_biquad =
                        std::make_shared<audio::EqualizerBiquadOperation>(sample_rate, center_freq, gain, Q);
                      THROW_IF_ERROR(equalizer_biquad->ValidateParams());
                      return equalizer_biquad;
                    }));
                }));

PYBIND_REGISTER(FadeShape, 0, ([](const py::module *m) {
                  (void)py::enum_<FadeShape>(*m, "FadeShape", py::arithmetic())
                    .value("DE_FADE_SHAPE_LINEAR", FadeShape::kLinear)
                    .value("DE_FADE_SHAPE_EXPONENTIAL", FadeShape::kExponential)
                    .value("DE_FADE_SHAPE_LOGARITHMIC", FadeShape::kLogarithmic)
                    .value("DE_FADE_SHAPE_QUARTER_SINE", FadeShape::kQuarterSine)
                    .value("DE_FADE_SHAPE_HALF_SINE", FadeShape::kHalfSine)
                    .export_values();
                }));

PYBIND_REGISTER(FadeOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::FadeOperation, TensorOperation, std::shared_ptr<audio::FadeOperation>>(
                    *m, "FadeOperation")
                    .def(py::init([](int fade_in_len, int fade_out_len, FadeShape fade_shape) {
                      auto fade = std::make_shared<audio::FadeOperation>(fade_in_len, fade_out_len, fade_shape);
                      THROW_IF_ERROR(fade->ValidateParams());
                      return fade;
                    }));
                }));

PYBIND_REGISTER(FiltfiltOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::FiltfiltOperation, TensorOperation, std::shared_ptr<audio::FiltfiltOperation>>(
                      *m, "FiltfiltOperation")
                      .def(py::init([](const std::vector<float> &a_coeffs, std::vector<float> &b_coeffs, bool clamp) {
                        auto filtfilt = std::make_shared<audio::FiltfiltOperation>(a_coeffs, b_coeffs, clamp);
                        THROW_IF_ERROR(filtfilt->ValidateParams());
                        return filtfilt;
                      }));
                }));

PYBIND_REGISTER(Modulation, 0, ([](const py::module *m) {
                  (void)py::enum_<Modulation>(*m, "Modulation", py::arithmetic())
                    .value("DE_MODULATION_SINUSOIDAL", Modulation::kSinusoidal)
                    .value("DE_MODULATION_TRIANGULAR", Modulation::kTriangular)
                    .export_values();
                }));

PYBIND_REGISTER(Interpolation, 0, ([](const py::module *m) {
                  (void)py::enum_<Interpolation>(*m, "Interpolation", py::arithmetic())
                    .value("DE_INTERPOLATION_LINEAR", Interpolation::kLinear)
                    .value("DE_INTERPOLATION_QUADRATIC", Interpolation::kQuadratic)
                    .export_values();
                }));

PYBIND_REGISTER(FlangerOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::FlangerOperation, TensorOperation, std::shared_ptr<audio::FlangerOperation>>(
                    *m, "FlangerOperation")
                    .def(py::init([](int32_t sample_rate, float delay, float depth, float regen, float width,
                                     float speed, float phase, Modulation modulation, Interpolation interpolation) {
                      auto flanger = std::make_shared<audio::FlangerOperation>(sample_rate, delay, depth, regen, width,
                                                                               speed, phase, modulation, interpolation);
                      THROW_IF_ERROR(flanger->ValidateParams());
                      return flanger;
                    }));
                }));

PYBIND_REGISTER(
  FrequencyMaskingOperation, 1, ([](const py::module *m) {
    (void)
      py::class_<audio::FrequencyMaskingOperation, TensorOperation, std::shared_ptr<audio::FrequencyMaskingOperation>>(
        *m, "FrequencyMaskingOperation")
        .def(py::init([](bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value) {
          auto frequency_masking =
            std::make_shared<audio::FrequencyMaskingOperation>(iid_masks, frequency_mask_param, mask_start, mask_value);
          THROW_IF_ERROR(frequency_masking->ValidateParams());
          return frequency_masking;
        }));
  }));

PYBIND_REGISTER(GainOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::GainOperation, TensorOperation, std::shared_ptr<audio::GainOperation>>(
                    *m, "GainOperation")
                    .def(py::init([](float gain_db) {
                      auto gain = std::make_shared<audio::GainOperation>(gain_db);
                      THROW_IF_ERROR(gain->ValidateParams());
                      return gain;
                    }));
                }));

PYBIND_REGISTER(
  GriffinLimOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::GriffinLimOperation, TensorOperation, std::shared_ptr<audio::GriffinLimOperation>>(
      *m, "GriffinLimOperation")
      .def(py::init([](int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length, WindowType window_type,
                       float power, float momentum, int32_t length, bool rand_init) {
        auto griffin_lim = std::make_shared<audio::GriffinLimOperation>(
          n_fft, n_iter, win_length, hop_length, window_type, power, momentum, length, rand_init);
        THROW_IF_ERROR(griffin_lim->ValidateParams());
        return griffin_lim;
      }));
  }));

PYBIND_REGISTER(
  HighpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::HighpassBiquadOperation, TensorOperation, std::shared_ptr<audio::HighpassBiquadOperation>>(
      *m, "HighpassBiquadOperation")
      .def(py::init([](float sample_rate, float cutoff_freq, float Q) {
        auto highpass_biquad = std::make_shared<audio::HighpassBiquadOperation>(sample_rate, cutoff_freq, Q);
        THROW_IF_ERROR(highpass_biquad->ValidateParams());
        return highpass_biquad;
      }));
  }));

PYBIND_REGISTER(InverseMelScaleOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::InverseMelScaleOperation, TensorOperation,
                                   std::shared_ptr<audio::InverseMelScaleOperation>>(*m, "InverseMelScaleOperation")
                    .def(py::init([](int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min, float f_max,
                                     int32_t max_iter, float tolerance_loss, float tolerance_change,
                                     const py::dict &sgdargs, NormType norm, MelType mel_type) {
                      auto inverse_mel_scale = std::make_shared<audio::InverseMelScaleOperation>(
                        n_stft, n_mels, sample_rate, f_min, f_max, max_iter, tolerance_loss, tolerance_change,
                        toStringFloatMap(sgdargs), norm, mel_type);
                      THROW_IF_ERROR(inverse_mel_scale->ValidateParams());
                      return inverse_mel_scale;
                    }));
                }));

PYBIND_REGISTER(LFilterOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::LFilterOperation, TensorOperation, std::shared_ptr<audio::LFilterOperation>>(
                    *m, "LFilterOperation")
                    .def(py::init([](std::vector<float> a_coeffs, std::vector<float> b_coeffs, bool clamp) {
                      auto lfilter = std::make_shared<audio::LFilterOperation>(a_coeffs, b_coeffs, clamp);
                      THROW_IF_ERROR(lfilter->ValidateParams());
                      return lfilter;
                    }));
                }));

PYBIND_REGISTER(
  LowpassBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::LowpassBiquadOperation, TensorOperation, std::shared_ptr<audio::LowpassBiquadOperation>>(
      *m, "LowpassBiquadOperation")
      .def(py::init([](int sample_rate, float cutoff_freq, float Q) {
        auto lowpass_biquad = std::make_shared<audio::LowpassBiquadOperation>(sample_rate, cutoff_freq, Q);
        THROW_IF_ERROR(lowpass_biquad->ValidateParams());
        return lowpass_biquad;
      }));
  }));

PYBIND_REGISTER(MagphaseOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::MagphaseOperation, TensorOperation, std::shared_ptr<audio::MagphaseOperation>>(
                      *m, "MagphaseOperation")
                      .def(py::init([](float power) {
                        auto magphase = std::make_shared<audio::MagphaseOperation>(power);
                        THROW_IF_ERROR(magphase->ValidateParams());
                        return magphase;
                      }));
                }));

PYBIND_REGISTER(MaskAlongAxisIIDOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::MaskAlongAxisIIDOperation, TensorOperation,
                                   std::shared_ptr<audio::MaskAlongAxisIIDOperation>>(*m, "MaskAlongAxisIIDOperation")
                    .def(py::init([](int32_t mask_param, float mask_value, int32_t axis) {
                      auto mask_along_axis_iid =
                        std::make_shared<audio::MaskAlongAxisIIDOperation>(mask_param, mask_value, axis);
                      THROW_IF_ERROR(mask_along_axis_iid->ValidateParams());
                      return mask_along_axis_iid;
                    }));
                }));

PYBIND_REGISTER(
  MaskAlongAxisOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::MaskAlongAxisOperation, TensorOperation, std::shared_ptr<audio::MaskAlongAxisOperation>>(
      *m, "MaskAlongAxisOperation")
      .def(py::init([](int32_t mask_start, int32_t mask_width, float mask_value, int32_t axis) {
        auto mask_along_axis =
          std::make_shared<audio::MaskAlongAxisOperation>(mask_start, mask_width, mask_value, axis);
        THROW_IF_ERROR(mask_along_axis->ValidateParams());
        return mask_along_axis;
      }));
  }));

PYBIND_REGISTER(MelScaleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::MelScaleOperation, TensorOperation, std::shared_ptr<audio::MelScaleOperation>>(
                      *m, "MelScaleOperation")
                      .def(py::init([](int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft,
                                       NormType norm, MelType mel_type) {
                        auto mel_scale = std::make_shared<audio::MelScaleOperation>(n_mels, sample_rate, f_min, f_max,
                                                                                    n_stft, norm, mel_type);
                        THROW_IF_ERROR(mel_scale->ValidateParams());
                        return mel_scale;
                      }));
                }));

PYBIND_REGISTER(
  MuLawDecodingOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::MuLawDecodingOperation, TensorOperation, std::shared_ptr<audio::MuLawDecodingOperation>>(
      *m, "MuLawDecodingOperation")
      .def(py::init([](int32_t quantization_channels) {
        auto mu_law_decoding = std::make_shared<audio::MuLawDecodingOperation>(quantization_channels);
        THROW_IF_ERROR(mu_law_decoding->ValidateParams());
        return mu_law_decoding;
      }));
  }));

PYBIND_REGISTER(
  MuLawEncodingOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::MuLawEncodingOperation, TensorOperation, std::shared_ptr<audio::MuLawEncodingOperation>>(
      *m, "MuLawEncodingOperation")
      .def(py::init([](int32_t quantization_channels) {
        auto mu_law_encoding = std::make_shared<audio::MuLawEncodingOperation>(quantization_channels);
        THROW_IF_ERROR(mu_law_encoding->ValidateParams());
        return mu_law_encoding;
      }));
  }));

PYBIND_REGISTER(OverdriveOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::OverdriveOperation, TensorOperation, std::shared_ptr<audio::OverdriveOperation>>(
                      *m, "OverdriveOperation")
                      .def(py::init([](float gain, float color) {
                        auto overdrive = std::make_shared<audio::OverdriveOperation>(gain, color);
                        THROW_IF_ERROR(overdrive->ValidateParams());
                        return overdrive;
                      }));
                }));

PYBIND_REGISTER(PhaserOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::PhaserOperation, TensorOperation, std::shared_ptr<audio::PhaserOperation>>(
                    *m, "PhaserOperation")
                    .def(py::init([](int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay,
                                     float mod_speed, bool sinusoidal) {
                      auto phaser = std::make_shared<audio::PhaserOperation>(sample_rate, gain_in, gain_out, delay_ms,
                                                                             decay, mod_speed, sinusoidal);
                      THROW_IF_ERROR(phaser->ValidateParams());
                      return phaser;
                    }));
                }));

PYBIND_REGISTER(
  PhaseVocoderOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::PhaseVocoderOperation, TensorOperation, std::shared_ptr<audio::PhaseVocoderOperation>>(
      *m, "PhaseVocoderOperation")
      .def(py::init([](float rate, const std::shared_ptr<Tensor> &phase_advance) {
        auto phase_vocoder = std::make_shared<audio::PhaseVocoderOperation>(rate, phase_advance);
        THROW_IF_ERROR(phase_vocoder->ValidateParams());
        return phase_vocoder;
      }));
  }));

PYBIND_REGISTER(ResampleMethod, 0, ([](const py::module *m) {
                  (void)py::enum_<ResampleMethod>(*m, "ResampleMethod", py::arithmetic())
                    .value("DE_RESAMPLE_SINC_INTERPOLATION", ResampleMethod::kSincInterpolation)
                    .value("DE_RESAMPLE_KAISER_WINDOW", ResampleMethod::kKaiserWindow)
                    .export_values();
                }));

PYBIND_REGISTER(ResampleOperation, 1, ([](const py::module *m) {
                  (void)
                    py::class_<audio::ResampleOperation, TensorOperation, std::shared_ptr<audio::ResampleOperation>>(
                      *m, "ResampleOperation")
                      .def(py::init([](float orig_freq, float new_freq, ResampleMethod resample_method,
                                       int32_t lowpass_filter_width, float rolloff, float beta) {
                        auto resample = std::make_shared<audio::ResampleOperation>(orig_freq, new_freq, resample_method,
                                                                                   lowpass_filter_width, rolloff, beta);
                        THROW_IF_ERROR(resample->ValidateParams());
                        return resample;
                      }));
                }));

PYBIND_REGISTER(
  RiaaBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::RiaaBiquadOperation, TensorOperation, std::shared_ptr<audio::RiaaBiquadOperation>>(
      *m, "RiaaBiquadOperation")
      .def(py::init([](int32_t sample_rate) {
        auto riaa_biquad = std::make_shared<audio::RiaaBiquadOperation>(sample_rate);
        THROW_IF_ERROR(riaa_biquad->ValidateParams());
        return riaa_biquad;
      }));
  }));

PYBIND_REGISTER(SlidingWindowCmnOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::SlidingWindowCmnOperation, TensorOperation,
                                   std::shared_ptr<audio::SlidingWindowCmnOperation>>(*m, "SlidingWindowCmnOperation")
                    .def(py::init([](int32_t cmn_window, int32_t min_cmn_window, bool center, bool norm_vars) {
                      auto sliding_window_cmn = std::make_shared<audio::SlidingWindowCmnOperation>(
                        cmn_window, min_cmn_window, center, norm_vars);
                      THROW_IF_ERROR(sliding_window_cmn->ValidateParams());
                      return sliding_window_cmn;
                    }));
                }));

PYBIND_REGISTER(WindowType, 0, ([](const py::module *m) {
                  (void)py::enum_<WindowType>(*m, "WindowType", py::arithmetic())
                    .value("DE_WINDOW_TYPE_BARTLETT", WindowType::kBartlett)
                    .value("DE_WINDOW_TYPE_BLACKMAN", WindowType::kBlackman)
                    .value("DE_WINDOW_TYPE_HAMMING", WindowType::kHamming)
                    .value("DE_WINDOW_TYPE_HANN", WindowType::kHann)
                    .value("DE_WINDOW_TYPE_KAISER", WindowType::kKaiser)
                    .export_values();
                }));

PYBIND_REGISTER(
  SpectralCentroidOperation, 1, ([](const py::module *m) {
    (void)
      py::class_<audio::SpectralCentroidOperation, TensorOperation, std::shared_ptr<audio::SpectralCentroidOperation>>(
        *m, "SpectralCentroidOperation")
        .def(py::init([](int sample_rate, int n_fft, int win_length, int hop_length, int pad, WindowType window) {
          auto spectral_centroid =
            std::make_shared<audio::SpectralCentroidOperation>(sample_rate, n_fft, win_length, hop_length, pad, window);
          THROW_IF_ERROR(spectral_centroid->ValidateParams());
          return spectral_centroid;
        }));
  }));

PYBIND_REGISTER(
  SpectrogramOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::SpectrogramOperation, TensorOperation, std::shared_ptr<audio::SpectrogramOperation>>(
      *m, "SpectrogramOperation")
      .def(py::init([](int32_t n_fft, int32_t win_length, int32_t hop_length, int32_t pad, WindowType window,
                       float power, bool normalized, bool center, BorderType pad_mode, bool onesided) {
        auto spectrogram = std::make_shared<audio::SpectrogramOperation>(n_fft, win_length, hop_length, pad, window,
                                                                         power, normalized, center, pad_mode, onesided);
        THROW_IF_ERROR(spectrogram->ValidateParams());
        return spectrogram;
      }));
  }));

PYBIND_REGISTER(
  TimeMaskingOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::TimeMaskingOperation, TensorOperation, std::shared_ptr<audio::TimeMaskingOperation>>(
      *m, "TimeMaskingOperation")
      .def(py::init([](bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value) {
        auto time_masking =
          std::make_shared<audio::TimeMaskingOperation>(iid_masks, time_mask_param, mask_start, mask_value);
        THROW_IF_ERROR(time_masking->ValidateParams());
        return time_masking;
      }));
  }));

PYBIND_REGISTER(
  TimeStretchOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::TimeStretchOperation, TensorOperation, std::shared_ptr<audio::TimeStretchOperation>>(
      *m, "TimeStretchOperation")
      .def(py::init([](float hop_length, int n_freq, float fixed_rate) {
        auto timestretch = std::make_shared<audio::TimeStretchOperation>(hop_length, n_freq, fixed_rate);
        THROW_IF_ERROR(timestretch->ValidateParams());
        return timestretch;
      }));
  }));

PYBIND_REGISTER(
  TrebleBiquadOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::TrebleBiquadOperation, TensorOperation, std::shared_ptr<audio::TrebleBiquadOperation>>(
      *m, "TrebleBiquadOperation")
      .def(py::init([](int32_t sample_rate, float gain, float central_freq, float Q) {
        auto treble_biquad = std::make_shared<audio::TrebleBiquadOperation>(sample_rate, gain, central_freq, Q);
        THROW_IF_ERROR(treble_biquad->ValidateParams());
        return treble_biquad;
      }));
  }));

PYBIND_REGISTER(VadOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::VadOperation, TensorOperation, std::shared_ptr<audio::VadOperation>>(
                    *m, "VadOperation")
                    .def(py::init([](int32_t sample_rate, float trigger_level, float trigger_time, float search_time,
                                     float allowed_gap, float pre_trigger_time, float boot_time, float noise_up_time,
                                     float noise_down_time, float noise_reduction_amount, float measure_freq,
                                     float measure_duration, float measure_smooth_time, float hp_filter_freq,
                                     float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq) {
                      auto vad = std::make_shared<audio::VadOperation>(
                        sample_rate, trigger_level, trigger_time, search_time, allowed_gap, pre_trigger_time, boot_time,
                        noise_up_time, noise_down_time, noise_reduction_amount, measure_freq, measure_duration,
                        measure_smooth_time, hp_filter_freq, lp_filter_freq, hp_lifter_freq, lp_lifter_freq);
                      THROW_IF_ERROR(vad->ValidateParams());
                      return vad;
                    }));
                }));

PYBIND_REGISTER(VolOperation, 1, ([](const py::module *m) {
                  (void)py::class_<audio::VolOperation, TensorOperation, std::shared_ptr<audio::VolOperation>>(
                    *m, "VolOperation")
                    .def(py::init([](float gain, GainType gain_type) {
                      auto vol = std::make_shared<audio::VolOperation>(gain, gain_type);
                      THROW_IF_ERROR(vol->ValidateParams());
                      return vol;
                    }));
                }));

PYBIND_REGISTER(GainType, 0, ([](const py::module *m) {
                  (void)py::enum_<GainType>(*m, "GainType", py::arithmetic())
                    .value("DE_GAIN_TYPE_AMPLITUDE", GainType::kAmplitude)
                    .value("DE_GAIN_TYPE_POWER", GainType::kPower)
                    .value("DE_GAIN_TYPE_DB", GainType::kDb)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
