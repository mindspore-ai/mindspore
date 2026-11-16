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

#include "minddata/dataset/include/dataset/audio.h"

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
#include "minddata/dataset/audio/transform/validators.h"
#include "minddata/dataset/audio/kernels/audio_utils.h"

namespace mindspore {
namespace dataset {
namespace audio {
// AllpassBiquad Transform Operation.
struct AllpassBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
};

AllpassBiquad::AllpassBiquad(int32_t sample_rate, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q)) {}

std::shared_ptr<TensorOperation> AllpassBiquad::Parse() {
  return std::make_shared<AllpassBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_);
}

// BandBiquad Transform Operation.
struct BandBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q, bool noise)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), noise_(noise) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool noise_;
};

BandBiquad::BandBiquad(int32_t sample_rate, float central_freq, float Q, bool noise)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q, noise)) {}

std::shared_ptr<TensorOperation> BandBiquad::Parse() {
  return std::make_shared<BandBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_, data_->noise_);
}

// BandpassBiquad Transform Operation.
struct BandpassBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q), const_skirt_gain_(const_skirt_gain) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
  bool const_skirt_gain_;
};

BandpassBiquad::BandpassBiquad(int32_t sample_rate, float central_freq, float Q, bool const_skirt_gain)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q, const_skirt_gain)) {}

std::shared_ptr<TensorOperation> BandpassBiquad::Parse() {
  return std::make_shared<BandpassBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_,
                                                   data_->const_skirt_gain_);
}

// BandrejectBiquad Transform Operation.
struct BandrejectBiquad::Data {
  Data(int32_t sample_rate, float central_freq, float Q)
      : sample_rate_(sample_rate), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float central_freq_;
  float Q_;
};

BandrejectBiquad::BandrejectBiquad(int32_t sample_rate, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, central_freq, Q)) {}

std::shared_ptr<TensorOperation> BandrejectBiquad::Parse() {
  return std::make_shared<BandrejectBiquadOperation>(data_->sample_rate_, data_->central_freq_, data_->Q_);
}

// BassBiquad Transform Operation.
struct BassBiquad::Data {
  Data(int32_t sample_rate, float gain, float central_freq, float Q)
      : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};

BassBiquad::BassBiquad(int32_t sample_rate, float gain, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, gain, central_freq, Q)) {}

std::shared_ptr<TensorOperation> BassBiquad::Parse() {
  return std::make_shared<BassBiquadOperation>(data_->sample_rate_, data_->gain_, data_->central_freq_, data_->Q_);
}

// Biquad Transform Operation.
struct Biquad::Data {
  Data(float b0, float b1, float b2, float a0, float a1, float a2)
      : b0_(b0), b1_(b1), b2_(b2), a0_(a0), a1_(a1), a2_(a2) {}
  float b0_;
  float b1_;
  float b2_;
  float a0_;
  float a1_;
  float a2_;
};

Biquad::Biquad(float b0, float b1, float b2, float a0, float a1, float a2)
    : data_(std::make_shared<Data>(b0, b1, b2, a0, a1, a2)) {}

std::shared_ptr<TensorOperation> Biquad::Parse() {
  return std::make_shared<BiquadOperation>(data_->b0_, data_->b1_, data_->b2_, data_->a0_, data_->a1_, data_->a1_);
}

// Contrast Transform Operation.
struct Contrast::Data {
  explicit Data(float enhancement_amount) : enhancement_amount_(enhancement_amount) {}
  float enhancement_amount_;
};

Contrast::Contrast(float enhancement_amount) : data_(std::make_shared<Data>(enhancement_amount)) {}

std::shared_ptr<TensorOperation> Contrast::Parse() {
  return std::make_shared<ContrastOperation>(data_->enhancement_amount_);
}

// DCShift Transform Operation.
struct DCShift::Data {
  Data(float shift, float limiter_gain) : shift_(shift), limiter_gain_(limiter_gain) {}
  float shift_;
  float limiter_gain_;
};

DCShift::DCShift(float shift) : data_(std::make_shared<Data>(shift, shift)) {}

DCShift::DCShift(float shift, float limiter_gain) : data_(std::make_shared<Data>(shift, limiter_gain)) {}

std::shared_ptr<TensorOperation> DCShift::Parse() {
  return std::make_shared<DCShiftOperation>(data_->shift_, data_->limiter_gain_);
}

Status CreateDct(mindspore::MSTensor *output, int32_t n_mfcc, int32_t n_mels, NormMode norm) {
  RETURN_UNEXPECTED_IF_NULL(output);
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mfcc", n_mfcc));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("CreateDct", "n_mels", n_mels));

  std::shared_ptr<dataset::Tensor> dct;
  RETURN_IF_NOT_OK(Dct(&dct, n_mfcc, n_mels, norm));
  CHECK_FAIL_RETURN_UNEXPECTED(dct->HasData(), "CreateDct: get an empty tensor with shape " + dct->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(dct));
  return Status::OK();
}

// DeemphBiquad Transform Operation.
struct DeemphBiquad::Data {
  explicit Data(int32_t sample_rate) : sample_rate_(sample_rate) {}
  int32_t sample_rate_;
};

DeemphBiquad::DeemphBiquad(int32_t sample_rate) : data_(std::make_shared<Data>(sample_rate)) {}

std::shared_ptr<TensorOperation> DeemphBiquad::Parse() {
  return std::make_shared<DeemphBiquadOperation>(data_->sample_rate_);
}

// Dither Transform Operation.
struct Dither::Data {
  Data(DensityFunction density_function, bool noise_shaping)
      : density_function_(density_function), noise_shaping_(noise_shaping) {}
  DensityFunction density_function_;
  bool noise_shaping_;
};

Dither::Dither(DensityFunction density_function, bool noise_shaping)
    : data_(std::make_shared<Data>(density_function, noise_shaping)) {}

std::shared_ptr<TensorOperation> Dither::Parse() {
  return std::make_shared<DitherOperation>(data_->density_function_, data_->noise_shaping_);
}

// EqualizerBiquad Transform Operation.
struct EqualizerBiquad::Data {
  Data(int32_t sample_rate, float center_freq, float gain, float Q)
      : sample_rate_(sample_rate), center_freq_(center_freq), gain_(gain), Q_(Q) {}
  int32_t sample_rate_;
  float center_freq_;
  float gain_;
  float Q_;
};

EqualizerBiquad::EqualizerBiquad(int32_t sample_rate, float center_freq, float gain, float Q)
    : data_(std::make_shared<Data>(sample_rate, center_freq, gain, Q)) {}

std::shared_ptr<TensorOperation> EqualizerBiquad::Parse() {
  return std::make_shared<EqualizerBiquadOperation>(data_->sample_rate_, data_->center_freq_, data_->gain_, data_->Q_);
}

// Fade Transform Operation.
struct Fade::Data {
  Data(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
      : fade_in_len_(fade_in_len), fade_out_len_(fade_out_len), fade_shape_(fade_shape) {}
  int32_t fade_in_len_;
  int32_t fade_out_len_;
  FadeShape fade_shape_;
};

Fade::Fade(int32_t fade_in_len, int32_t fade_out_len, FadeShape fade_shape)
    : data_(std::make_shared<Data>(fade_in_len, fade_out_len, fade_shape)) {}

std::shared_ptr<TensorOperation> Fade::Parse() {
  return std::make_shared<FadeOperation>(data_->fade_in_len_, data_->fade_out_len_, data_->fade_shape_);
}

// Filtfilt Transform Operation.
struct Filtfilt::Data {
  Data(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};

Filtfilt::Filtfilt(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : data_(std::make_shared<Data>(a_coeffs, b_coeffs, clamp)) {}

std::shared_ptr<TensorOperation> Filtfilt::Parse() {
  return std::make_shared<FiltfiltOperation>(data_->a_coeffs_, data_->b_coeffs_, data_->clamp_);
}

// Flanger Transform Operation.
struct Flanger::Data {
  Data(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
       Modulation modulation, Interpolation interpolation)
      : sample_rate_(sample_rate),
        delay_(delay),
        depth_(depth),
        regen_(regen),
        width_(width),
        speed_(speed),
        phase_(phase),
        modulation_(modulation),
        interpolation_(interpolation) {}
  int32_t sample_rate_;
  float delay_;
  float depth_;
  float regen_;
  float width_;
  float speed_;
  float phase_;
  Modulation modulation_;
  Interpolation interpolation_;
};

Flanger::Flanger(int32_t sample_rate, float delay, float depth, float regen, float width, float speed, float phase,
                 Modulation modulation, Interpolation interpolation)
    : data_(std::make_shared<Data>(sample_rate, delay, depth, regen, width, speed, phase, modulation, interpolation)) {}

std::shared_ptr<TensorOperation> Flanger::Parse() {
  return std::make_shared<FlangerOperation>(data_->sample_rate_, data_->delay_, data_->depth_, data_->regen_,
                                            data_->width_, data_->speed_, data_->phase_, data_->modulation_,
                                            data_->interpolation_);
}

// Gain Transform Operation.
struct Gain::Data {
  explicit Data(float gain_db) : gain_db_(gain_db) {}
  float gain_db_;
};

Gain::Gain(float gain_db) : data_(std::make_shared<Data>(gain_db)) {}

std::shared_ptr<TensorOperation> Gain::Parse() { return std::make_shared<GainOperation>(data_->gain_db_); }

// HighpassBiquad Transform Operation.
struct HighpassBiquad::Data {
  Data(int32_t sample_rate, float cutoff_freq, float Q) : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};

HighpassBiquad::HighpassBiquad(int32_t sample_rate, float cutoff_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, cutoff_freq, Q)) {}

std::shared_ptr<TensorOperation> HighpassBiquad::Parse() {
  return std::make_shared<HighpassBiquadOperation>(data_->sample_rate_, data_->cutoff_freq_, data_->Q_);
}

// LFilter Transform Operation.
struct LFilter::Data {
  Data(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
      : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}
  std::vector<float> a_coeffs_;
  std::vector<float> b_coeffs_;
  bool clamp_;
};

LFilter::LFilter(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : data_(std::make_shared<Data>(a_coeffs, b_coeffs, clamp)) {}

std::shared_ptr<TensorOperation> LFilter::Parse() {
  return std::make_shared<LFilterOperation>(data_->a_coeffs_, data_->b_coeffs_, data_->clamp_);
}

// LowpassBiquad Transform Operation.
struct LowpassBiquad::Data {
  Data(int32_t sample_rate, float cutoff_freq, float Q) : sample_rate_(sample_rate), cutoff_freq_(cutoff_freq), Q_(Q) {}
  int32_t sample_rate_;
  float cutoff_freq_;
  float Q_;
};

LowpassBiquad::LowpassBiquad(int32_t sample_rate, float cutoff_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, cutoff_freq, Q)) {}

std::shared_ptr<TensorOperation> LowpassBiquad::Parse() {
  return std::make_shared<LowpassBiquadOperation>(data_->sample_rate_, data_->cutoff_freq_, data_->Q_);
}

// MelscaleFbanks Function.
Status MelscaleFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_mels, int32_t sample_rate,
                      NormType norm, MelType mel_type) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "MelscaleFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "MelscaleFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0,
                               "MelscaleFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_mels > 0,
                               "MelscaleFbanks: n_mels must be greater than 0, got: " + std::to_string(n_mels));
  CHECK_FAIL_RETURN_UNEXPECTED(
    sample_rate > 0, "MelscaleFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "MelscaleFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateFbanks<float>(&fb, n_freqs, f_min, f_max, n_mels, sample_rate, norm, mel_type));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(),
                               "MelscaleFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

// MuLawDecoding Transform Operation.
struct MuLawDecoding::Data {
  explicit Data(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}
  int32_t quantization_channels_;
};

MuLawDecoding::MuLawDecoding(int32_t quantization_channels) : data_(std::make_shared<Data>(quantization_channels)) {}

std::shared_ptr<TensorOperation> MuLawDecoding::Parse() {
  return std::make_shared<MuLawDecodingOperation>(data_->quantization_channels_);
}

// MuLawEncoding Transform Operation.
struct MuLawEncoding::Data {
  explicit Data(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}
  int32_t quantization_channels_;
};

MuLawEncoding::MuLawEncoding(int32_t quantization_channels) : data_(std::make_shared<Data>(quantization_channels)) {}

std::shared_ptr<TensorOperation> MuLawEncoding::Parse() {
  return std::make_shared<MuLawEncodingOperation>(data_->quantization_channels_);
}

// LinearFbanks Function.
Status LinearFbanks(MSTensor *output, int32_t n_freqs, float f_min, float f_max, int32_t n_filter,
                    int32_t sample_rate) {
  RETURN_UNEXPECTED_IF_NULL(output);
  CHECK_FAIL_RETURN_UNEXPECTED(n_freqs > 0,
                               "LinearFbanks: n_freqs must be greater than 0, got: " + std::to_string(n_freqs));

  CHECK_FAIL_RETURN_UNEXPECTED(f_min >= 0, "LinearFbanks: f_min must be non negative, got: " + std::to_string(f_min));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > 0, "LinearFbanks: f_max must be greater than 0, got: " + std::to_string(f_max));
  CHECK_FAIL_RETURN_UNEXPECTED(n_filter > 0,
                               "LinearFbanks: n_filter must be greater than 0, got: " + std::to_string(n_filter));
  CHECK_FAIL_RETURN_UNEXPECTED(sample_rate > 0,
                               "LinearFbanks: sample_rate must be greater than 0, got: " + std::to_string(sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(f_max > f_min, "LinearFbanks: f_max must be greater than f_min, got: f_min = " +
                                                std::to_string(f_min) + ", while f_max = " + std::to_string(f_max));
  std::shared_ptr<dataset::Tensor> fb;
  RETURN_IF_NOT_OK(CreateLinearFbanks(&fb, n_freqs, f_min, f_max, n_filter, sample_rate));
  CHECK_FAIL_RETURN_UNEXPECTED(fb->HasData(), "LinearFbanks: get an empty tensor with shape " + fb->shape().ToString());
  *output = mindspore::MSTensor(std::make_shared<DETensor>(fb));
  return Status::OK();
}

// Overdrive Transform Operation.
struct Overdrive::Data {
  Data(float gain, float color) : gain_(gain), color_(color) {}
  float gain_;
  float color_;
};

Overdrive::Overdrive(float gain, float color) : data_(std::make_shared<Data>(gain, color)) {}

std::shared_ptr<TensorOperation> Overdrive::Parse() {
  return std::make_shared<OverdriveOperation>(data_->gain_, data_->color_);
}

// Phaser Transform Operation.
struct Phaser::Data {
  Data(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
       bool sinusoidal)
      : sample_rate_(sample_rate),
        gain_in_(gain_in),
        gain_out_(gain_out),
        delay_ms_(delay_ms),
        decay_(decay),
        mod_speed_(mod_speed),
        sinusoidal_(sinusoidal) {}
  int32_t sample_rate_;
  float gain_in_;
  float gain_out_;
  float delay_ms_;
  float decay_;
  float mod_speed_;
  bool sinusoidal_;
};

Phaser::Phaser(int32_t sample_rate, float gain_in, float gain_out, float delay_ms, float decay, float mod_speed,
               bool sinusoidal)
    : data_(std::make_shared<Data>(sample_rate, gain_in, gain_out, delay_ms, decay, mod_speed, sinusoidal)) {}

std::shared_ptr<TensorOperation> Phaser::Parse() {
  return std::make_shared<PhaserOperation>(data_->sample_rate_, data_->gain_in_, data_->gain_out_, data_->delay_ms_,
                                           data_->decay_, data_->mod_speed_, data_->sinusoidal_);
}

// RiaaBiquad Transform Operation.
struct RiaaBiquad::Data {
  explicit Data(int32_t sample_rate) : sample_rate_(sample_rate) {}
  int32_t sample_rate_;
};

RiaaBiquad::RiaaBiquad(int32_t sample_rate) : data_(std::make_shared<Data>(sample_rate)) {}

std::shared_ptr<TensorOperation> RiaaBiquad::Parse() {
  return std::make_shared<RiaaBiquadOperation>(data_->sample_rate_);
}

// TrebleBiquad Transform Operation.
struct TrebleBiquad::Data {
  Data(int32_t sample_rate, float gain, float central_freq, float Q)
      : sample_rate_(sample_rate), gain_(gain), central_freq_(central_freq), Q_(Q) {}
  int32_t sample_rate_;
  float gain_;
  float central_freq_;
  float Q_;
};

TrebleBiquad::TrebleBiquad(int32_t sample_rate, float gain, float central_freq, float Q)
    : data_(std::make_shared<Data>(sample_rate, gain, central_freq, Q)) {}

std::shared_ptr<TensorOperation> TrebleBiquad::Parse() {
  return std::make_shared<TrebleBiquadOperation>(data_->sample_rate_, data_->gain_, data_->central_freq_, data_->Q_);
}

// Vad Transform Operation.
struct Vad::Data {
  Data(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
       float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
       float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
       float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
      : sample_rate_(sample_rate),
        trigger_level_(trigger_level),
        trigger_time_(trigger_time),
        search_time_(search_time),
        allowed_gap_(allowed_gap),
        pre_trigger_time_(pre_trigger_time),
        boot_time_(boot_time),
        noise_up_time_(noise_up_time),
        noise_down_time_(noise_down_time),
        noise_reduction_amount_(noise_reduction_amount),
        measure_freq_(measure_freq),
        measure_duration_(measure_duration),
        measure_smooth_time_(measure_smooth_time),
        hp_filter_freq_(hp_filter_freq),
        lp_filter_freq_(lp_filter_freq),
        hp_lifter_freq_(hp_lifter_freq),
        lp_lifter_freq_(lp_lifter_freq) {}
  int32_t sample_rate_;
  float trigger_level_;
  float trigger_time_;
  float search_time_;
  float allowed_gap_;
  float pre_trigger_time_;
  float boot_time_;
  float noise_up_time_;
  float noise_down_time_;
  float noise_reduction_amount_;
  float measure_freq_;
  float measure_duration_;
  float measure_smooth_time_;
  float hp_filter_freq_;
  float lp_filter_freq_;
  float hp_lifter_freq_;
  float lp_lifter_freq_;
};

Vad::Vad(int32_t sample_rate, float trigger_level, float trigger_time, float search_time, float allowed_gap,
         float pre_trigger_time, float boot_time, float noise_up_time, float noise_down_time,
         float noise_reduction_amount, float measure_freq, float measure_duration, float measure_smooth_time,
         float hp_filter_freq, float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
    : data_(std::make_shared<Data>(sample_rate, trigger_level, trigger_time, search_time, allowed_gap, pre_trigger_time,
                                   boot_time, noise_up_time, noise_down_time, noise_reduction_amount, measure_freq,
                                   measure_duration, measure_smooth_time, hp_filter_freq, lp_filter_freq,
                                   hp_lifter_freq, lp_lifter_freq)) {}

std::shared_ptr<TensorOperation> Vad::Parse() {
  return std::make_shared<VadOperation>(
    data_->sample_rate_, data_->trigger_level_, data_->trigger_time_, data_->search_time_, data_->allowed_gap_,
    data_->pre_trigger_time_, data_->boot_time_, data_->noise_up_time_, data_->noise_down_time_,
    data_->noise_reduction_amount_, data_->measure_freq_, data_->measure_duration_, data_->measure_smooth_time_,
    data_->hp_filter_freq_, data_->lp_filter_freq_, data_->hp_lifter_freq_, data_->lp_lifter_freq_);
}

// Vol Transform Operation.
struct Vol::Data {
  Data(float gain, GainType gain_type) : gain_(gain), gain_type_(gain_type) {}
  float gain_;
  GainType gain_type_;
};

Vol::Vol(float gain, GainType gain_type) : data_(std::make_shared<Data>(gain, gain_type)) {}

std::shared_ptr<TensorOperation> Vol::Parse() {
  return std::make_shared<VolOperation>(data_->gain_, data_->gain_type_);
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
