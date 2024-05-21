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

#include "minddata/dataset/kernels/ir/vision/solarize_ir.h"

#include "minddata/dataset/kernels/image/solarize_op.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_solarize_op.h"
#endif
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// SolarizeOperation
SolarizeOperation::SolarizeOperation(const std::vector<float> &threshold, const std::string &device_target)
    : threshold_(threshold), device_target_(device_target) {}

SolarizeOperation::~SolarizeOperation() = default;

Status SolarizeOperation::ValidateParams() {
  constexpr size_t kThresholdSize = 2;
  constexpr float kThresholdMax = 255.0;

  if (threshold_.size() != kThresholdSize) {
    std::string err_msg =
      "Solarize: threshold must be a vector of two values, got: " + std::to_string(threshold_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (float threshold_value : threshold_) {
    if (threshold_value < 0 || threshold_value > kThresholdMax) {
      std::string err_msg = "Solarize: threshold has to be between 0 and 255, got:" + std::to_string(threshold_value);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  if (threshold_[0] > threshold_[1]) {
    std::string err_msg = "Solarize: threshold must be passed in a (min, max) format";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Solarize: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SolarizeOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<SolarizeOp> tensor_op = std::make_shared<SolarizeOp>(threshold_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppSolarizeOp> dvpp_tensor_op = std::make_shared<DvppSolarizeOp>(threshold_);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "Solarize: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status SolarizeOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["threshold"] = threshold_;
  (*out_json)["device_target"] = device_target_;
  return Status::OK();
}

Status SolarizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "threshold", kSolarizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kSolarizeOperation));
  std::vector<float> threshold = op_params["threshold"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::SolarizeOperation>(threshold, device_target);
  return Status::OK();
}

MapTargetDevice SolarizeOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Solarize: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
