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
#include "minddata/dataset/kernels/ir/vision/adjust_brightness_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/adjust_brightness_op.h"
#endif
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_adjust_brightness.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AdjustBrightnessOperation
AdjustBrightnessOperation::AdjustBrightnessOperation(float brightness_factor, const std::string &device_target)
    : brightness_factor_(brightness_factor), device_target_(device_target) {}

Status AdjustBrightnessOperation::ValidateParams() {
  // brightness_factor
  RETURN_IF_NOT_OK(ValidateFloatScalarNonNegative("AdjustBrightness", "brightness_factor", brightness_factor_));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "AdjustBrightness: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AdjustBrightnessOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<AdjustBrightnessOp> tensor_op = std::make_shared<AdjustBrightnessOp>(brightness_factor_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppAdjustBrightnessOp>(brightness_factor_);
#endif
  } else {
    MS_LOG(ERROR) << "AdjustBrightness: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status AdjustBrightnessOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["brightness_factor"] = brightness_factor_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status AdjustBrightnessOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "brightness_factor", kAdjustBrightnessOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kAdjustBrightnessOperation));
  float brightness_factor = op_params["brightness_factor"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::AdjustBrightnessOperation>(brightness_factor, device_target);
  return Status::OK();
}

MapTargetDevice AdjustBrightnessOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "AdjustBrightness: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
