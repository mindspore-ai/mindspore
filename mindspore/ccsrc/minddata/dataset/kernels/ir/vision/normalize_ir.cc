/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/normalize_ir.h"

#include "minddata/dataset/kernels/image/normalize_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// NormalizeOperation
NormalizeOperation::NormalizeOperation(const std::vector<float> &mean, const std::vector<float> &std)
    : mean_(mean), std_(std) {}

NormalizeOperation::~NormalizeOperation() = default;

std::string NormalizeOperation::Name() const { return kNormalizeOperation; }

Status NormalizeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorMeanStd("Normalize", mean_, std_));
  return Status::OK();
}

std::shared_ptr<TensorOp> NormalizeOperation::Build() { return std::make_shared<NormalizeOp>(mean_, std_); }

Status NormalizeOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["mean"] = mean_;
  args["std"] = std_;
  *out_json = args;
  return Status::OK();
}

Status NormalizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("mean") != op_params.end(), "Fail to find mean");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("std") != op_params.end(), "Fail to find std");
  std::vector<float> mean = op_params["mean"];
  std::vector<float> std = op_params["std"];
  *operation = std::make_shared<vision::NormalizeOperation>(mean, std);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
