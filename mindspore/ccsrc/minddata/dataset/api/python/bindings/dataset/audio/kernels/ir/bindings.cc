/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/ir/kernels/contrast_ir.h"
#include "minddata/dataset/audio/ir/kernels/dc_shift_ir.h"
#include "minddata/dataset/audio/ir/kernels/deemph_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/equalizer_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/fade_ir.h"
#include "minddata/dataset/audio/ir/kernels/frequency_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/highpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/lfilter_ir.h"
#include "minddata/dataset/audio/ir/kernels/lowpass_biquad_ir.h"
#include "minddata/dataset/audio/ir/kernels/magphase_ir.h"
#include "minddata/dataset/audio/ir/kernels/mu_law_decoding_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_stretch_ir.h"
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
                    .value("DE_SCALETYPE_MAGNITUDE", ScaleType::kMagnitude)
                    .value("DE_SCALETYPE_POWER", ScaleType::kPower)
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
                    .value("DE_FADESHAPE_LINEAR", FadeShape::kLinear)
                    .value("DE_FADESHAPE_EXPONENTIAL", FadeShape::kExponential)
                    .value("DE_FADESHAPE_LOGARITHMIC", FadeShape::kLogarithmic)
                    .value("DE_FADESHAPE_QUARTERSINE", FadeShape::kQuarterSine)
                    .value("DE_FADESHAPE_HALFSINE", FadeShape::kHalfSine)
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

PYBIND_REGISTER(
  MuLawDecodingOperation, 1, ([](const py::module *m) {
    (void)py::class_<audio::MuLawDecodingOperation, TensorOperation, std::shared_ptr<audio::MuLawDecodingOperation>>(
      *m, "MuLawDecodingOperation")
      .def(py::init([](int quantization_channels) {
        auto mu_law_decoding = std::make_shared<audio::MuLawDecodingOperation>(quantization_channels);
        THROW_IF_ERROR(mu_law_decoding->ValidateParams());
        return mu_law_decoding;
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
                    .value("DE_GAINTYPE_AMPLITUDE", GainType::kAmplitude)
                    .value("DE_GAINTYPE_POWER", GainType::kPower)
                    .value("DE_GAINTYPE_DB", GainType::kDb)
                    .export_values();
                }));
}  // namespace dataset
}  // namespace mindspore
