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

#include "minddata/dataset/audio/ir/kernels/detect_pitch_frequency_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/detect_pitch_frequency_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// DetectPitchFrequencyOperation
DetectPitchFrequencyOperation::DetectPitchFrequencyOperation(int32_t sample_rate, float frame_time, int32_t win_length,
                                                             int32_t freq_low, int32_t freq_high)
    : sample_rate_(sample_rate),
      frame_time_(frame_time),
      win_length_(win_length),
      freq_low_(freq_low),
      freq_high_(freq_high) {}

Status DetectPitchFrequencyOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateScalarNotZero("DetectPitchFrequency", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("DetectPitchFrequency", "frame_time", frame_time_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("DetectPitchFrequency", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("DetectPitchFrequency", "freq_low", freq_low_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("DetectPitchFrequency", "freq_high", freq_high_));
  return Status::OK();
}

std::shared_ptr<TensorOp> DetectPitchFrequencyOperation::Build() {
  std::shared_ptr<DetectPitchFrequencyOp> tensor_op =
    std::make_shared<DetectPitchFrequencyOp>(sample_rate_, frame_time_, win_length_, freq_low_, freq_high_);
  return tensor_op;
}

Status DetectPitchFrequencyOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["sample_rate"] = sample_rate_;
  args["frame_time"] = frame_time_;
  args["win_length"] = win_length_;
  args["freq_low"] = freq_low_;
  args["freq_high"] = freq_high_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
