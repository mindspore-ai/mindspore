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

#include "minddata/dataset/kernels/ir/vision/random_equalize_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_equalize_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomEqualizeOperation
RandomEqualizeOperation::RandomEqualizeOperation(float prob) : TensorOperation(true), probability_(prob) {}

RandomEqualizeOperation::~RandomEqualizeOperation() = default;

std::string RandomEqualizeOperation::Name() const { return kRandomEqualizeOperation; }

Status RandomEqualizeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateProbability("RandomEqualize", probability_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomEqualizeOperation::Build() {
  std::shared_ptr<RandomEqualizeOp> tensor_op = std::make_shared<RandomEqualizeOp>(probability_);
  return tensor_op;
}

Status RandomEqualizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["prob"] = probability_;
  return Status::OK();
}

Status RandomEqualizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "prob", kRandomEqualizeOperation));
  float prob = op_params["prob"];
  *operation = std::make_shared<vision::RandomEqualizeOperation>(prob);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
