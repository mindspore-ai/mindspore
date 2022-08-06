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
#include "minddata/dataset/kernels/ir/vision/adjust_brightness_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_brightness_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustBrightnessOperation
AdjustBrightnessOperation::AdjustBrightnessOperation(float brightness_factor) : brightness_factor_(brightness_factor) {}

Status AdjustBrightnessOperation::ValidateParams() {
  // brightness_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustBrightness", "brightness_factor", brightness_factor_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustBrightnessOperation::Build() {
  std::shared_ptr<AdjustBrightnessOp> tensor_op = std::make_shared<AdjustBrightnessOp>(brightness_factor_);
  return tensor_op;
}

Status AdjustBrightnessOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["brightness_factor"] = brightness_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustBrightnessOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "brightness_factor", kAdjustBrightnessOperation));
  float brightness_factor = op_params["brightness_factor"];
  *operation = std::make_shared<vision::AdjustBrightnessOperation>(brightness_factor);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
