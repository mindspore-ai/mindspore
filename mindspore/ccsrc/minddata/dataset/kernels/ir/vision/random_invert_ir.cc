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
#include "minddata/dataset/kernels/ir/vision/random_invert_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_invert_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomInvertOperation
RandomInvertOperation::RandomInvertOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomInvertOperation::~RandomInvertOperation() = default;

std::string RandomInvertOperation::Name() const { return kRandomInvertOperation; }

Status RandomInvertOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomInvert", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomInvertOperation::Build() {
  std::shared_ptr<RandomInvertOp> tensor_op = std::make_shared<RandomInvertOp>(probability_);
  return tensor_op;
}

Status RandomInvertOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomInvertOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomInvertOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomInvertOperation>(prob);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
