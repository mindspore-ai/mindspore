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
#include "minddata/dataset/audio/ir/kernels/frequency_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_masking_ir.h"
#include "minddata/dataset/audio/ir/kernels/time_stretch_ir.h"

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

// FrequencyMasking Transform Operation.
struct FrequencyMasking::Data {
  Data(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, double mask_value)
      : iid_masks_(iid_masks),
        frequency_mask_param_(frequency_mask_param),
        mask_start_(mask_start),
        mask_value_(mask_value) {}
  int32_t frequency_mask_param_;
  int32_t mask_start_;
  bool iid_masks_;
  double mask_value_;
};

FrequencyMasking::FrequencyMasking(bool iid_masks, int32_t frequency_mask_param, int32_t mask_start, double mask_value)
    : data_(std::make_shared<Data>(iid_masks, frequency_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> FrequencyMasking::Parse() {
  return std::make_shared<FrequencyMaskingOperation>(data_->iid_masks_, data_->frequency_mask_param_,
                                                     data_->mask_start_, data_->mask_value_);
}

// TimeMasking Transform Operation.
struct TimeMasking::Data {
  Data(bool iid_masks, int64_t time_mask_param, int64_t mask_start, double mask_value)
      : iid_masks_(iid_masks), time_mask_param_(time_mask_param), mask_start_(mask_start), mask_value_(mask_value) {}
  int64_t time_mask_param_;
  int64_t mask_start_;
  bool iid_masks_;
  double mask_value_;
};

TimeMasking::TimeMasking(bool iid_masks, int64_t time_mask_param, int64_t mask_start, double mask_value)
    : data_(std::make_shared<Data>(iid_masks, time_mask_param, mask_start, mask_value)) {}

std::shared_ptr<TensorOperation> TimeMasking::Parse() {
  return std::make_shared<TimeMaskingOperation>(data_->iid_masks_, data_->time_mask_param_, data_->mask_start_,
                                                data_->mask_value_);
}

// TimeStretch Transform Operation.
struct TimeStretch::Data {
  explicit Data(float hop_length, int n_freq, float fixed_rate)
      : hop_length_(hop_length), n_freq_(n_freq), fixed_rate_(fixed_rate) {}
  float hop_length_;
  int n_freq_;
  float fixed_rate_;
};

TimeStretch::TimeStretch(float hop_length, int n_freq, float fixed_rate)
    : data_(std::make_shared<Data>(hop_length, n_freq, fixed_rate)) {}

std::shared_ptr<TensorOperation> TimeStretch::Parse() {
  return std::make_shared<TimeStretchOperation>(data_->hop_length_, data_->n_freq_, data_->fixed_rate_);
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
