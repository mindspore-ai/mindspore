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
#include "minddata/dataset/kernels/ir/vision/adjust_contrast_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_contrast_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustContrastOperation
AdjustContrastOperation::AdjustContrastOperation(float contrast_factor) : contrast_factor_(contrast_factor) {}

Status AdjustContrastOperation::ValidateParams() {
  // contrast_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustContrast", "contrast_factor", contrast_factor_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustContrastOperation::Build() {
  std::shared_ptr<AdjustContrastOp> tensor_op = std::make_shared<AdjustContrastOp>(contrast_factor_);
  return tensor_op;
}

Status AdjustContrastOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["contrast_factor"] = contrast_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustContrastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "contrast_factor", kAdjustContrastOperation));
  float contrast_factor = op_params["contrast_factor"];
  *operation = std::make_shared<vision::AdjustContrastOperation>(contrast_factor);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
