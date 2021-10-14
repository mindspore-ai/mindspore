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

#include "minddata/dataset/include/dataset/audio.h"

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

// AmplitudeToDB Transform Operation.
struct AmplitudeToDB::Data {
  Data(ScaleType stype, float ref_value, float amin, float top_db)
      : stype_(stype), ref_value_(ref_value), amin_(amin), top_db_(top_db) {}
  ScaleType stype_;
  float ref_value_;
  float amin_;
  float top_db_;
};

AmplitudeToDB::AmplitudeToDB(ScaleType stype, float ref_value, float amin, float top_db)
    : data_(std::make_shared<Data>(stype, ref_value, amin, top_db)) {}

std::shared_ptr<TensorOperation> AmplitudeToDB::Parse() {
  return std::make_shared<AmplitudeToDBOperation>(data_->stype_, data_->ref_value_, data_->amin_, data_->top_db_);
}

// Angle Transform Operation.
Angle::Angle() {}

std::shared_ptr<TensorOperation> Angle::Parse() { return std::make_shared<AngleOperation>(); }
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

// ComplexNorm Transform Operation.
struct ComplexNorm::Data {
  explicit Data(float power) : power_(power) {}
  float power_;
};

ComplexNorm::ComplexNorm(float power) : data_(std::make_shared<Data>(power)) {}

std::shared_ptr<TensorOperation> ComplexNorm::Parse() { return std::make_shared<ComplexNormOperation>(data_->power_); }

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

// DeemphBiquad Transform Operation.
struct DeemphBiquad::Data {
  explicit Data(int32_t sample_rate) : sample_rate_(sample_rate) {}
  int32_t sample_rate_;
};

DeemphBiquad::DeemphBiquad(int32_t sample_rate) : data_(std::make_shared<Data>(sample_rate)) {}

std::shared_ptr<TensorOperation> DeemphBiquad::Parse() {
  return std::make_shared<DeemphBiquadOperation>(data_->sample_rate_);
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

// FrequencyMasking Transform Operation.
struct FrequencyMasking::Data {
  Data(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value)
      : iid_masks_(iid_masks),
        frequency_mask_param_(frequency_mask_param),
        mask_start_(mask_start),
        mask_value_(mask_value) {}
  bool iid_masks_;
  int32_t frequency_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};

FrequencyMasking::FrequencyMasking(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, float mask_value)
    : data_(std::make_shared<Data>(iid_masks, frequency_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> FrequencyMasking::Parse() {
  return std::make_shared<FrequencyMaskingOperation>(data_->iid_masks_, data_->frequency_mask_param_,
                                                     data_->mask_start_, data_->mask_value_);
}

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

LFilter::LFilter(std::vector<float> a_coeffs, std::vector<float> b_coeffs, bool clamp)
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

// Magphase Transform Operation.
struct Magphase::Data {
  explicit Data(float power) : power_(power) {}
  float power_;
};

Magphase::Magphase(float power) : data_(std::make_shared<Data>(power)) {}

std::shared_ptr<TensorOperation> Magphase::Parse() { return std::make_shared<MagphaseOperation>(data_->power_); }

// MuLawDecoding Transform Operation.
struct MuLawDecoding::Data {
  explicit Data(int32_t quantization_channels) : quantization_channels_(quantization_channels) {}
  int32_t quantization_channels_;
};

MuLawDecoding::MuLawDecoding(int32_t quantization_channels) : data_(std::make_shared<Data>(quantization_channels)) {}

std::shared_ptr<TensorOperation> MuLawDecoding::Parse() {
  return std::make_shared<MuLawDecodingOperation>(data_->quantization_channels_);
}

// TimeMasking Transform Operation.
struct TimeMasking::Data {
  Data(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
      : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {}
  bool iid_masks_;
  int32_t time_mask_param_;
  int32_t mask_start_;
  float mask_value_;
};

TimeMasking::TimeMasking(bool iid_masks, int32_t time_mask_param, int32_t mask_start, float mask_value)
    : data_(std::make_shared<Data>(iid_masks, time_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> TimeMasking::Parse() {
  return std::make_shared<TimeMaskingOperation>(data_->iid_masks_, data_->time_mask_param_, data_->mask_start_,
                                                data_->mask_value_);
}

// TimeStretch Transform Operation.
struct TimeStretch::Data {
  explicit Data(float hop_length, int32_t n_freq, float fixed_rate)
      : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}
  float hop_length_;
  int32_t n_freq_;
  float fixed_rate_;
};

TimeStretch::TimeStretch(float hop_length, int32_t n_freq, float fixed_rate)
    : data_(std::make_shared<Data>(hop_length, n_freq, fixed_rate)) {}

std::shared_ptr<TensorOperation> TimeStretch::Parse() {
  return std::make_shared<TimeStretchOperation>(data_->hop_length_, data_->n_freq_, data_->fixed_rate_);
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
