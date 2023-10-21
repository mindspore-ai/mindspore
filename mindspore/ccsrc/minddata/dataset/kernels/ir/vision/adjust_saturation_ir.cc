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
#include "minddata/dataset/kernels/ir/vision/adjust_saturation_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_saturation_op.h"
#endif
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_adjust_saturation.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustSaturationOperation
AdjustSaturationOperation::AdjustSaturationOperation(float saturation_factor, const std::string &device_target)
    : saturation_factor_(saturation_factor), device_target_(device_target) {}

Status AdjustSaturationOperation::ValidateParams() {
  // saturation_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustSaturation", "saturation_factor", saturation_factor_));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "AdjustSaturation: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustSaturationOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<AdjustSaturationOp> tensor_op = std::make_shared<AdjustSaturationOp>(saturation_factor_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppAdjustSaturationOp>(saturation_factor_);
#endif
  } else {
    MS_LOG(ERROR) << "AdjustSaturation: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status AdjustSaturationOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["saturation_factor"] = saturation_factor_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status AdjustSaturationOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "saturation_factor", kAdjustSaturationOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kAdjustSaturationOperation));
  float saturation_factor = op_params["saturation_factor"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::AdjustSaturationOperation>(saturation_factor, device_target);
  return Status::OK();
}

MapTargetDevice AdjustSaturationOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "AdjustSaturation: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
