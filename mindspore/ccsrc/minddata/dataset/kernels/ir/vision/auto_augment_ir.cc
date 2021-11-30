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
#include "minddata/dataset/kernels/ir/vision/auto_augment_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/auto_augment_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AutoAugmentOperation
AutoAugmentOperation::AutoAugmentOperation(AutoAugmentPolicy policy, InterpolationMode interpolation,
                                           const std::vector<uint8_t> &fill_value)
    : policy_(policy), interpolation_(interpolation), fill_value_(fill_value) {}

AutoAugmentOperation::~AutoAugmentOperation() = default;

std::string AutoAugmentOperation::Name() const { return kAutoAugmentOperation; }

Status AutoAugmentOperation::ValidateParams() {
  if (policy_ != AutoAugmentPolicy::kImageNet && policy_ != AutoAugmentPolicy::kCifar10 &&
      policy_ != AutoAugmentPolicy::kSVHN) {
    std::string err_msg = "AutoAugment: Invalid AutoAugmentPolicy, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea) {
    std::string err_msg = "AutoAugment: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("AutoAugment", fill_value_));
  return Status::OK();
}

std::shared_ptr<TensorOp> AutoAugmentOperation::Build() {
  std::shared_ptr<AutoAugmentOp> tensor_op = std::make_shared<AutoAugmentOp>(policy_, interpolation_, fill_value_);
  return tensor_op;
}

Status AutoAugmentOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["policy"] = policy_;
  args["interpolation"] = interpolation_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status AutoAugmentOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "policy", kAutoAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kAutoAugmentOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kAutoAugmentOperation));
  AutoAugmentPolicy policy = op_params["policy"];
  InterpolationMode interpolation = op_params["interpolation"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::AutoAugmentOperation>(policy, interpolation, fill_value);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
