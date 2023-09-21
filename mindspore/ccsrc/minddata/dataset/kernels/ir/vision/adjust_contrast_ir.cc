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
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_adjust_contrast.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustContrastOperation
AdjustContrastOperation::AdjustContrastOperation(float contrast_factor, const std::string &device_target)
    : contrast_factor_(contrast_factor), device_target_(device_target) {}

Status AdjustContrastOperation::ValidateParams() {
  // contrast_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustContrast", "contrast_factor", contrast_factor_));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "AdjustContrast: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustContrastOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<AdjustContrastOp> tensor_op = std::make_shared<AdjustContrastOp>(contrast_factor_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppAdjustContrastOp>(contrast_factor_);
#endif
  } else {
    MS_LOG(ERROR) << "AdjustContrast: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status AdjustContrastOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["contrast_factor"] = contrast_factor_;
  args["device_target"] = contrast_factor_;
  *out_json = args;
  return Status::OK();
}

Status AdjustContrastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "contrast_factor", kAdjustContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kAdjustContrastOperation));
  float contrast_factor = op_params["contrast_factor"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::AdjustContrastOperation>(contrast_factor, device_target);
  return Status::OK();
}

MapTargetDevice AdjustContrastOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "AdjustContrast: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
