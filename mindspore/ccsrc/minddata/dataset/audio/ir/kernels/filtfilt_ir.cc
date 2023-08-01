/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/audio/ir/kernels/filtfilt_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/filtfilt_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// FiltfiltOperation
FiltfiltOperation::FiltfiltOperation(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

Status FiltfiltOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("filtfilt", "a_coeffs", a_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("filtfilt", "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorSameSize("filtfilt", "a_coeffs", a_coeffs_, "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("filtfilt", "a_coeffs[0]", a_coeffs_[0]));
  return Status::OK();
}

std::shared_ptr<TensorOp> FiltfiltOperation::Build() {
  std::shared_ptr<FiltfiltOp> tensor_op = std::make_shared<FiltfiltOp>(a_coeffs_, b_coeffs_, clamp_);
  return tensor_op;
}

Status FiltfiltOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["a_coeffs"] = a_coeffs_;
  args["b_coeffs"] = b_coeffs_;
  args["clamp"] = clamp_;
  *out_json = args;
  return Status::OK();
}
}  // namespace audio
}  // namespace dataset
}  // namespace mindspore
