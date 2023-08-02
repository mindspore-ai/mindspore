/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/random_lighting_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_lighting_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomLightingOperation.
RandomLightingOperation::RandomLightingOperation(float alpha) : TensorOperation(true), alpha_(alpha) {}

RandomLightingOperation::~RandomLightingOperation() = default;

std::string RandomLightingOperation::Name() const { return kRandomLightingOperation; }

Status RandomLightingOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("RandomLighting", "alpha", alpha_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomLightingOperation::Build() {
  std::shared_ptr<RandomLightingOp> tensor_op = std::make_shared<RandomLightingOp>(alpha_);
  return tensor_op;
}

Status RandomLightingOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["alpha"] = alpha_;
  *out_json = args;
  return Status::OK();
}

Status RandomLightingOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "alpha", kRandomLightingOperation));
  float alpha = op_params["alpha"];
  *operation = std::make_shared<vision::RandomLightingOperation>(alpha);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
