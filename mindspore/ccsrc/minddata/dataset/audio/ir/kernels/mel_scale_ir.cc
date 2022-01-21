/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/audio/ir/kernels/mel_scale_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/mel_scale_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace audio {
// MelScale
MelScaleOperation::MelScaleOperation(int32_t n_mels, int32_t sample_rate, float f_min, float f_max, int32_t n_stft,
                                     NormType norm, MelType mel_type)
    : n_mels_(n_mels),
      sample_rate_(sample_rate),
      f_min_(f_min),
      f_max_(f_max),
      n_stft_(n_stft),
      norm_(norm),
      mel_type_(mel_type) {}

MelScaleOperation::~MelScaleOperation() = default;

std::string MelScaleOperation::Name() const { return kMelScaleOperation; }

Status MelScaleOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelScale", "n_mels", n_mels_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelScale", "sample_rate", sample_rate_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("MelScale", "n_stft", n_stft_));
  CHECK_FAIL_RETURN_UNEXPECTED(n_stft_ != 1, "MelScale: n_stft can not be 1.");
  if (f_max_ != 0) {
    RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("MelScale", "f_max", f_max_));
    CHECK_FAIL_RETURN_UNEXPECTED(f_min_ < f_max_, "MelScale: f_max must be greater than f_min.");
  } else {
    float half = 0.5;
    CHECK_FAIL_RETURN_UNEXPECTED(f_min_ < (sample_rate_ * half),
                                 "MelScale: f_min must be less than sample_rate / 2 when f_max is 0.");
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> MelScaleOperation::Build() {
  float f_max = f_max_ == 0 ? static_cast<float>(sample_rate_) / 2 : f_max_;
  std::shared_ptr<MelScaleOp> tensor_op =
    std::make_shared<MelScaleOp>(n_mels_, sample_rate_, f_min_, f_max, n_stft_, norm_, mel_type_);
  return tensor_op;
}

Status MelScaleOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["n_mels"] = n_mels_;
  args["sample_rate"] = sample_rate_;
  args["f_min"] = f_min_;
  args["f_max"] = f_max_;
  args["n_stft"] = n_stft_;
  args["norm"] = norm_;
  args["mel_type"] = mel_type_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
