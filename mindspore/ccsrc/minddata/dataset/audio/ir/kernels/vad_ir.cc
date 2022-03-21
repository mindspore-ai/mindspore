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

#include "minddata/dataset/audio/ir/kernels/vad_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/vad_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// Vad
VadOperation::VadOperation(int32_t sample_rate, float trigger_level, float trigger_time, float search_time,
                           float allowed_gap, float pre_trigger_time, float boot_time, float noise_up_time,
                           float noise_down_time, float noise_reduction_amount, float measure_freq,
                           float measure_duration, float measure_smooth_time, float hp_filter_freq,
                           float lp_filter_freq, float hp_lifter_freq, float lp_lifter_freq)
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

VadOperation::~VadOperation() = default;

std::string VadOperation::Name() const { return kVadOperation; }

Status VadOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Vad", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "trigger_time", trigger_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "search_time", search_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "allowed_gap", allowed_gap_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "pre_trigger_time", pre_trigger_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "boot_time", boot_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "noise_up_time", noise_up_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "noise_down_time", noise_down_time_));
  CHECK_FAIL_RETURN_UNEXPECTED(noise_down_time_ <= noise_up_time_,
                               "Vad: noise_up_time must be greater than or equal to noise_down_time.");
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "noise_reduction_amount", noise_reduction_amount_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vad", "measure_freq", measure_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "measure_duration", measure_duration_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("Vad", "measure_smooth_time", measure_smooth_time_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vad", "hp_filter_freq", hp_filter_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vad", "lp_filter_freq", lp_filter_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vad", "hp_lifter_freq", hp_lifter_freq_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("Vad", "lp_lifter_freq", lp_lifter_freq_));

  return Status::OK();
}

std::shared_ptr<TensorOp> VadOperation::Build() {
  float measure_duration = measure_duration_ == 0 ? 2.0 / measure_freq_ : measure_duration_;
  std::shared_ptr<VadOp> tensor_op = std::make_shared<VadOp>(
    sample_rate_, trigger_level_, trigger_time_, search_time_, allowed_gap_, pre_trigger_time_, boot_time_,
    noise_up_time_, noise_down_time_, noise_reduction_amount_, measure_freq_, measure_duration, measure_smooth_time_,
    hp_filter_freq_, lp_filter_freq_, hp_lifter_freq_, lp_lifter_freq_);
  return tensor_op;
}

Status VadOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["trigger_level"] = trigger_level_;
  args["trigger_time"] = trigger_time_;
  args["search_time"] = search_time_;
  args["allowed_gap"] = allowed_gap_;
  args["pre_trigger_time"] = pre_trigger_time_;
  args["boot_time"] = boot_time_;
  args["noise_up_time"] = noise_up_time_;
  args["noise_down_time"] = noise_down_time_;
  args["noise_reduction_amount"] = noise_reduction_amount_;
  args["measure_freq"] = measure_freq_;
  args["measure_duration"] = measure_duration_;
  args["measure_smooth_time"] = measure_smooth_time_;
  args["hp_filter_freq"] = hp_filter_freq_;
  args["lp_filter_freq"] = lp_filter_freq_;
  args["hp_lifter_freq"] = hp_lifter_freq_;
  args["lp_lifter_freq"] = lp_lifter_freq_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
