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

#include "minddata/dataset/kernels/ir/vision/erase_ir.h"

#include "minddata/dataset/audio/ir/validators.h"
#include "minddata/dataset/kernels/image/erase_op.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_erase_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// EraseOperation
EraseOperation::EraseOperation(int32_t top, int32_t left, int32_t height, int32_t width,
                               const std::vector<float> &value, bool inplace, const std::string &device_target)
    : top_(top),
      left_(left),
      height_(height),
      width_(width),
      value_(value),
      inplace_(inplace),
      device_target_(device_target) {}

EraseOperation::~EraseOperation() = default;

std::string EraseOperation::Name() const { return kEraseOperation; }

Status EraseOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Erase", "top", top_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("Erase", "left", left_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Erase", "height", height_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("Erase", "width", width_));
  constexpr float kValueMax = 255.0;
  constexpr float kValueMin = 0.;
  const size_t kMaxFillValueSize = 3;
  if (value_.empty() || (value_.size() != 1 && value_.size() != kMaxFillValueSize)) {
    std::string err_msg = "Erase: value expecting size 1 or 3, got value.size(): " + std::to_string(value_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  for (float val : value_) {
    if (val < kValueMin || val > kValueMax) {
      std::string err_msg = "Erase: value has to be between 0. and 255., got:" + std::to_string(val);
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Erase: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> EraseOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<EraseOp> tensor_op = std::make_shared<EraseOp>(top_, left_, height_, width_, value_, inplace_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::vector<float> value_cast(value_.begin(), value_.end());
    std::shared_ptr<DvppEraseOp> dvpp_tensor_op =
      std::make_shared<DvppEraseOp>(top_, left_, height_, width_, value_cast);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "AdjustContrast: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status EraseOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["top"] = top_;
  args["left"] = left_;
  args["height"] = height_;
  args["width"] = width_;
  args["value"] = value_;
  args["inplace"] = inplace_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status EraseOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "top", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "left", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "height", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "width", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "value", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "inplace", kEraseOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kEraseOperation));

  int32_t top = op_params["top"];
  int32_t left = op_params["left"];
  int32_t height = op_params["height"];
  int32_t width = op_params["width"];
  std::vector<float> value = op_params["value"];
  bool inplace = op_params["inplace"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::EraseOperation>(top, left, height, width, value, inplace, device_target);
  return Status::OK();
}

MapTargetDevice EraseOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Erase: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
