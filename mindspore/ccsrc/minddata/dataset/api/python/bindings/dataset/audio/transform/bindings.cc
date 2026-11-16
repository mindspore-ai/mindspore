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

#include "minddata/dataset/audio/transform/allpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/band_biquad_ir.h"
#include "minddata/dataset/audio/transform/bandpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/bandreject_biquad_ir.h"
#include "minddata/dataset/audio/transform/bass_biquad_ir.h"
#include "minddata/dataset/audio/transform/biquad_ir.h"
#include "minddata/dataset/audio/transform/contrast_ir.h"
#include "minddata/dataset/audio/transform/dc_shift_ir.h"
#include "minddata/dataset/audio/transform/deemph_biquad_ir.h"
#include "minddata/dataset/audio/transform/dither_ir.h"
#include "minddata/dataset/audio/transform/equalizer_biquad_ir.h"
#include "minddata/dataset/audio/transform/fade_ir.h"
#include "minddata/dataset/audio/transform/filtfilt_ir.h"
#include "minddata/dataset/audio/transform/flanger_ir.h"
#include "minddata/dataset/audio/transform/gain_ir.h"
#include "minddata/dataset/audio/transform/highpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/lfilter_ir.h"
#include "minddata/dataset/audio/transform/lowpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/mu_law_decoding_ir.h"
#include "minddata/dataset/audio/transform/mu_law_encoding_ir.h"
#include "minddata/dataset/audio/transform/overdrive_ir.h"
#include "minddata/dataset/audio/transform/phaser_ir.h"
#include "minddata/dataset/audio/transform/riaa_biquad_ir.h"
#include "minddata/dataset/audio/transform/treble_biquad_ir.h"
#include "minddata/dataset/audio/transform/vad_ir.h"
#include "minddata/dataset/audio/transform/vol_ir.h"

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

PYBIND_REGISTER(ScaleType, 0, ([](const py::module *m) {
                  (void)py::enum_<ScaleType>(*m, "ScaleType", py::arithmetic())
                    .value("DE_SCALE_TYPE_MAGNITUDE", ScaleType::kMagnitude)
                    .value("DE_SCALE_TYPE_POWER", ScaleType::kPower)
                    .export_values();
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

PYBIND_REGISTER(ResampleMethod, 0, ([](const py::module *m) {
                  (void)py::enum_<ResampleMethod>(*m, "ResampleMethod", py::arithmetic())
                    .value("DE_RESAMPLE_SINC_INTERPOLATION", ResampleMethod::kSincInterpolation)
                    .value("DE_RESAMPLE_KAISER_WINDOW", ResampleMethod::kKaiserWindow)
                    .export_values();
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
