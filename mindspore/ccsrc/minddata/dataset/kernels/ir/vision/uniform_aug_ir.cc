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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/uniform_aug_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/engine/serdes.h"
#include "minddata/dataset/kernels/image/uniform_aug_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// UniformAugOperation
UniformAugOperation::UniformAugOperation(const std::vector<std::shared_ptr<TensorOperation>> &transforms,
                                         int32_t num_ops)
    : transforms_(transforms), num_ops_(num_ops) {}

UniformAugOperation::~UniformAugOperation() = default;

std::string UniformAugOperation::Name() const { return kUniformAugOperation; }

Status UniformAugOperation::ValidateParams() {
  // transforms
  RETURN_IF_NOT_OK(ValidateVectorTransforms("UniformAug", transforms_));
  if (num_ops_ > transforms_.size()) {
    std::string err_msg =
      "UniformAug: num_ops must be less than or equal to transforms size, but got: " + std::to_string(num_ops_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  // num_ops
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("UniformAug", "num_ops", num_ops_));
  return Status::OK();
}

std::shared_ptr<TensorOp> UniformAugOperation::Build() {
  std::vector<std::shared_ptr<TensorOp>> tensor_ops;
  (void)std::transform(
    transforms_.begin(), transforms_.end(), std::back_inserter(tensor_ops),
    [](const std::shared_ptr<TensorOperation> &op) -> std::shared_ptr<TensorOp> { return op->Build(); });
  std::shared_ptr<UniformAugOp> tensor_op = std::make_shared<UniformAugOp>(tensor_ops, num_ops_);
  return tensor_op;
}

Status UniformAugOperation::to_json(nlohmann::json *out_json) {
  CHECK_FAIL_RETURN_UNEXPECTED(out_json != nullptr, "parameter out_json is nullptr");
  nlohmann::json args;
  std::vector<nlohmann::json> transforms;
  for (auto op : transforms_) {
    nlohmann::json op_item, op_args;
    RETURN_IF_NOT_OK(op->to_json(&op_args));
    op_item["tensor_op_params"] = op_args;
    op_item["tensor_op_name"] = op->Name();
    transforms.push_back(op_item);
  }
  args["transforms"] = transforms;
  args["num_ops"] = num_ops_;
  *out_json = args;
  return Status::OK();
}

Status UniformAugOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("transforms") != op_params.end(), "Failed to find transforms");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("num_ops") != op_params.end(), "Failed to find num_ops");
  std::vector<std::shared_ptr<TensorOperation>> transforms = {};
  RETURN_IF_NOT_OK(Serdes::ConstructTensorOps(op_params["transforms"], &transforms));
  int32_t num_ops = op_params["num_ops"];
  *operation = std::make_shared<vision::UniformAugOperation>(transforms, num_ops);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
