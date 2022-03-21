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

#include "minddata/dataset/audio/ir/kernels/griffin_lim_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/griffin_lim_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// GriffinLim
GriffinLimOperation::GriffinLimOperation(int32_t n_fft, int32_t n_iter, int32_t win_length, int32_t hop_length,
                                         WindowType window_type, float power, float momentum, int32_t length,
                                         bool rand_init)
    : n_fft_(n_fft),
      n_iter_(n_iter),
      win_length_(win_length),
      hop_length_(hop_length),
      window_type_(window_type),
      power_(power),
      momentum_(momentum),
      length_(length),
      rand_init_(rand_init) {}

GriffinLimOperation::~GriffinLimOperation() = default;

std::string GriffinLimOperation::Name() const { return kGriffinLimOperation; }

Status GriffinLimOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("GriffinLim", "n_fft", n_fft_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("GriffinLim", "n_iter", n_iter_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("GriffinLim", "win_length", win_length_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("GriffinLim", "hop_length", hop_length_));
  RETURN_IF_NOT_OK(ValidateFloatScalarPositive("GriffinLim", "power", power_));
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("GriffinLim", "momentum", momentum_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("GriffinLim", "length", length_));
  if (length_ != 0 && n_fft_ >= length_) {
    std::string err_msg = "GriffinLim: n_fft must be less than length.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    momentum_ < 1,
    "GriffinLim: momentum equal to or greater than 1 can be unstable, but got: " + std::to_string(momentum_));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(momentum_ >= 0,
                                 "GriffinLim: momentum can not be less than 0, but got: " + std::to_string(momentum_));
  CHECK_FAIL_RETURN_SYNTAX_ERROR(win_length_ <= n_fft_,
                                 "GriffinLim: win_length must be less than or equal to n_fft, but got win_length: " +
                                   std::to_string(win_length_) + ", n_fft: " + std::to_string(n_fft_));
  return Status::OK();
}

std::shared_ptr<TensorOp> GriffinLimOperation::Build() {
  int32_t win_length = (win_length_ == 0) ? n_fft_ : win_length_;
  int32_t hop_length = (hop_length_ == 0) ? win_length / 2 : hop_length_;
  float momentum = momentum_ / (1 + momentum_);
  std::shared_ptr<GriffinLimOp> tensor_op = std::make_shared<GriffinLimOp>(
    n_fft_, n_iter_, win_length, hop_length, window_type_, power_, momentum, length_, rand_init_);
  return tensor_op;
}

Status GriffinLimOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["n_fft"] = n_fft_;
  args["n_iter"] = n_iter_;
  args["win_length"] = win_length_;
  args["hop_length"] = hop_length_;
  args["window_type"] = window_type_;
  args["power"] = power_;
  args["momentum"] = momentum_;
  args["length"] = length_;
  args["rand_init"] = rand_init_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
