/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/ir/vision/auto_contrast_ir.h"

#include <algorithm>

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/auto_contrast_op.h"
#endif
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_auto_contrast_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// AutoContrastOperation
AutoContrastOperation::AutoContrastOperation(float cutoff, const std::vector<uint32_t> &ignore,
                                             const std::string &device_target)
    : cutoff_(cutoff), ignore_(ignore), device_target_(device_target) {}

AutoContrastOperation::~AutoContrastOperation() = default;

std::string AutoContrastOperation::Name() const { return kAutoContrastOperation; }

Status AutoContrastOperation::ValidateParams() {
  constexpr float kMaxCutOff = 100.0;
  if (cutoff_ < 0.0 || cutoff_ > kMaxCutOff) {
    std::string err_msg = "AutoContrast: 'cutoff' has to be between 0 and 100, got: " + std::to_string(cutoff_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  constexpr uint32_t kMaxIgnoreSize = 255;
  for (uint32_t single_ignore : ignore_) {
    if (single_ignore > kMaxIgnoreSize) {
      std::string err_msg =
        "AutoContrast: invalid size, 'ignore' has to be between 0 and 255, got: " + std::to_string(single_ignore);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "AutoContrast: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> AutoContrastOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<AutoContrastOp> tensor_op = std::make_shared<AutoContrastOp>(cutoff_, ignore_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::vector<float> dvpp_cutoff = {cutoff_, cutoff_};
    std::shared_ptr<DvppAutoContrastOp> dvpp_tensor_op = std::make_shared<DvppAutoContrastOp>(dvpp_cutoff, ignore_);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "AutoContrast: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status AutoContrastOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["cutoff"] = cutoff_;
  args["ignore"] = ignore_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status AutoContrastOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "cutoff", kAutoContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "ignore", kAutoContrastOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kAutoContrastOperation));
  float cutoff = op_params["cutoff"];
  std::vector<uint32_t> ignore = op_params["ignore"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::AutoContrastOperation>(cutoff, ignore, device_target);
  return Status::OK();
}

MapTargetDevice AutoContrastOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "AutoContrast: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
