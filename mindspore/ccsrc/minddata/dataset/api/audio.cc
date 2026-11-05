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
#include "minddata/dataset/audio/transform/deemph_biquad_ir.h"
#include "minddata/dataset/audio/transform/equalizer_biquad_ir.h"
#include "minddata/dataset/audio/transform/filtfilt_ir.h"
#include "minddata/dataset/audio/transform/highpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/lfilter_ir.h"
#include "minddata/dataset/audio/transform/lowpass_biquad_ir.h"
#include "minddata/dataset/audio/transform/riaa_biquad_ir.h"
#include "minddata/dataset/audio/transform/treble_biquad_ir.h"
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

}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
