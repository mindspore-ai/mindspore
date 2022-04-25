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
#include "minddata/dataset/audio/ir/kernels/inverse_mel_scale_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/inverse_mel_scale_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// InverseMelScale
InverseMelScaleOperation::InverseMelScaleOperation(int32_t n_stft, int32_t n_mels, int32_t sample_rate, float f_min,
                                                   float f_max, int32_t max_iter, float tolerance_loss,
                                                   float tolerance_change, const std::map<std::string, float> &sgdargs,
                                                   NormType norm, MelType mel_type)
    : n_stft_(n_stft),
      n_mels_(n_mels),
      sample_rate_(sample_rate),
      f_min_(f_min),
      f_max_(f_max),
      max_iter_(max_iter),
      tolerance_loss_(tolerance_loss),
      tolerance_change_(tolerance_change),
      sgdargs_(sgdargs),
      norm_(norm),
      mel_type_(mel_type) {
  sgd_lr_ = sgdargs_.find("sgd_lr") == sgdargs_.end() ? 0.1 : sgdargs_["sgd_lr"];
  constexpr float SGD_MOMENTUM_DEFAULT = 0.9;
  sgd_momentum_ = sgdargs_.find("sgd_momentum") == sgdargs_.end() ? SGD_MOMENTUM_DEFAULT : sgdargs_["sgd_momentum"];
}

InverseMelScaleOperation::~InverseMelScaleOperation() = default;

std::string InverseMelScaleOperation::Name() const { return kInverseMelScaleOperation; }

Status InverseMelScaleOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseMelScale", "n_mels", n_mels_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("InverseMelScale", "sample_rate", sample_rate_));
  CHECK_FAIL_RETURN_UNEXPECTED(n_stft_ != 1,
                               "InverseMelScale: n_stft can not be equal to 1, but got: " + std::to_string(n_stft_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("InverseMelScale", "f_max", f_max_));
  CHECK_FAIL_RETURN_UNEXPECTED(f_min_ < f_max_, "InverseMelScale: f_max must be greater than f_min.");

  // SGD params
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("InverseMelScale", "sgd_lr", sgd_lr_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("InverseMelScale", "sgd_momentum", sgd_momentum_));
  return Status::OK();
}

std::shared_ptr<TensorOp> InverseMelScaleOperation::Build() {
  std::shared_ptr<InverseMelScaleOp> tensor_op =
    std::make_shared<InverseMelScaleOp>(n_stft_, n_mels_, sample_rate_, f_min_, f_max_, max_iter_, tolerance_loss_,
                                        tolerance_change_, sgd_lr_, sgd_momentum_, norm_, mel_type_);
  return tensor_op;
}

Status InverseMelScaleOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["n_stft"] = n_stft_;
  args["n_mels"] = n_mels_;
  args["sample_rate"] = sample_rate_;
  args["f_min"] = f_min_;
  args["f_max"] = f_max_;
  args["max_iter"] = max_iter_;
  args["tolerance_loss"] = tolerance_loss_;
  args["tolerance_change"] = tolerance_change_;
  args["sgdargs"] = sgdargs_;
  args["norm"] = norm_;
  args["mel_type"] = mel_type_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
