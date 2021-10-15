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

#include "minddata/dataset/audio/ir/kernels/lfilter_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/audio/kernels/lfilter_op.h"

namespace mindspore {
namespace dataset {
namespace audio {
// LFilterOperation
LFilterOperation::LFilterOperation(const std::vector<float> &a_coeffs, const std::vector<float> &b_coeffs, bool clamp)
    : a_coeffs_(a_coeffs), b_coeffs_(b_coeffs), clamp_(clamp) {}

Status LFilterOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("lfilter", "a_coeffs", a_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorNotEmpty("lfilter", "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateVectorSameSize("lfilter", "a_coeffs", a_coeffs_, "b_coeffs", b_coeffs_));
  RETURN_IF_NOT_OK(ValidateScalarNotZero("lfilter", "a_coeffs[0]", a_coeffs_[0]));
  return Status::OK();
}

std::shared_ptr<TensorOp> LFilterOperation::Build() {
  std::shared_ptr<LFilterOp> tensor_op = std::make_shared<LFilterOp>(a_coeffs_, b_coeffs_, clamp_);
  return tensor_op;
}

Status LFilterOperation::to_json(nlohmann::json *out_json) {
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
