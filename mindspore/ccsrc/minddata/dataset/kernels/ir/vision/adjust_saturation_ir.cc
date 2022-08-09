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
#include "minddata/dataset/kernels/ir/vision/adjust_saturation_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_saturation_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustSaturationOperation
AdjustSaturationOperation::AdjustSaturationOperation(float saturation_factor) : saturation_factor_(saturation_factor) {}

Status AdjustSaturationOperation::ValidateParams() {
  // saturation_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustSaturation", "saturation_factor", saturation_factor_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustSaturationOperation::Build() {
  std::shared_ptr<AdjustSaturationOp> tensor_op = std::make_shared<AdjustSaturationOp>(saturation_factor_);
  return tensor_op;
}

Status AdjustSaturationOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["saturation_factor"] = saturation_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustSaturationOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "saturation_factor", kAdjustSaturationOperation));
  float saturation_factor = op_params["saturation_factor"];
  *operation = std::make_shared<vision::AdjustSaturationOperation>(saturation_factor);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
