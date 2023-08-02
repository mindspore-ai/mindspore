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
#include "minddata/dataset/kernels/ir/vision/adjust_sharpness_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/sharpness_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustSharpnessOperation
AdjustSharpnessOperation::AdjustSharpnessOperation(float sharpness_factor) : sharpness_factor_(sharpness_factor) {}

Status AdjustSharpnessOperation::ValidateParams() {
  // sharpness_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustSharpness", "sharpness_factor", sharpness_factor_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustSharpnessOperation::Build() {
  std::shared_ptr<SharpnessOp> tensor_op = std::make_shared<SharpnessOp>(sharpness_factor_);
  return tensor_op;
}

Status AdjustSharpnessOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["sharpness_factor"] = sharpness_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustSharpnessOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "sharpness_factor", kAdjustSharpnessOperation));
  float sharpness_factor = op_params["sharpness_factor"];
  *operation = std::make_shared<vision::AdjustSharpnessOperation>(sharpness_factor);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
