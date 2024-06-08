/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/kernels/ir/vision/rotate_ir.h"

#include "minddata/dataset/kernels/image/rotate_op.h"
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_rotate_op.h"
#endif

namespace mindspore {
namespace dataset {
namespace vision {
// RotateOperation
RotateOperation::RotateOperation(FixRotationAngle angle)
    : angle_id_(static_cast<uint64_t>(angle)),
      degrees_(0.0),
      interpolation_mode_(InterpolationMode::kLinear),
      expand_(false),
      center_({}),
      fill_value_({}) {}

RotateOperation::RotateOperation(float degrees, InterpolationMode resample, bool expand,
                                 const std::vector<float> &center, const std::vector<uint8_t> &fill_value,
                                 const std::string &device_target)
    : angle_id_(0),
      degrees_(degrees),
      interpolation_mode_(resample),
      expand_(expand),
      center_(center),
      fill_value_(fill_value),
      device_target_(device_target) {}

RotateOperation::~RotateOperation() = default;

std::string RotateOperation::Name() const { return kRotateOperation; }

Status RotateOperation::ValidateParams() {
#ifndef ENABLE_ANDROID
  // center
  constexpr auto kCenterSize = 2;
  if (!center_.empty() && center_.size() != kCenterSize) {
    std::string err_msg =
      "Rotate: center must be a vector of two values or empty, got: " + std::to_string(center_.size());
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // fill_value
  RETURN_IF_NOT_OK(ValidateVectorFillvalue("Rotate", fill_value_));

  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Rotate: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // interpolation
  if (device_target_ == "CPU") {
    if (interpolation_mode_ != InterpolationMode::kLinear &&
        interpolation_mode_ != InterpolationMode::kNearestNeighbour &&
        interpolation_mode_ != InterpolationMode::kCubic && interpolation_mode_ != InterpolationMode::kArea) {
      std::string err_msg = "Rotate: Invalid InterpolationMode, check input value of enum.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    if (interpolation_mode_ != InterpolationMode::kLinear &&
        interpolation_mode_ != InterpolationMode::kNearestNeighbour) {
      std::string err_msg = "DvppRotate: Invalid Interpolation mode, only support BILINEAR and NEAREST.";
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
#else
  if (angle_id_ < 1 || angle_id_ > 8) {
    std::string err_msg = "Rotate: angle_id must be in range of [1, 8], got: " + std::to_string(angle_id_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
#endif
  return Status::OK();
}

std::shared_ptr<TensorOp> RotateOperation::Build() {
#ifndef ENABLE_ANDROID
  uint8_t fill_r, fill_g, fill_b;
  fill_r = fill_value_[0];
  fill_g = fill_value_[0];
  fill_b = fill_value_[0];

  if (fill_value_.size() == 3) {
    fill_r = fill_value_[0];
    fill_g = fill_value_[1];
    fill_b = fill_value_[2];
  }

  if (device_target_ == "CPU") {
    std::shared_ptr<RotateOp> tensor_op =
      std::make_shared<RotateOp>(degrees_, interpolation_mode_, expand_, center_, fill_r, fill_g, fill_b);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppRotateOp> dvpp_tensor_op = std::make_shared<DvppRotateOp>(
      degrees_, interpolation_mode_, expand_, center_, fill_r, fill_g, fill_b);  // need change shenwei
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "Rotate: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
#else
  rotate_op_ = std::make_shared<RotateOp>(0);
  setAngle(angle_id_);
  return rotate_op_;
#endif
}

Status RotateOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
#ifndef ENABLE_ANDROID
  args["degree"] = degrees_;
  args["resample"] = interpolation_mode_;
  args["expand"] = expand_;
  args["center"] = center_;
  args["fill_value"] = fill_value_;
  args["device_target"] = device_target_;
#else
  args["angle_id"] = angle_id_;
#endif
  *out_json = args;
  return Status::OK();
}

Status RotateOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
#ifndef ENABLE_ANDROID
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "degree", kRotateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "resample", kRotateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "expand", kRotateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "center", kRotateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "fill_value", kRotateOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kRotateOperation));
  float degrees = op_params["degree"];
  auto resample = static_cast<InterpolationMode>(op_params["resample"]);
  bool expand = op_params["expand"];
  std::vector<float> center = op_params["center"];
  std::vector<uint8_t> fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::RotateOperation>(degrees, resample, expand, center, fill_value);
#else
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "angle_id", kRotateOperation));
  uint64_t angle_id = op_params["angle_id"];
  std::shared_ptr<RotateOperation> rotate_operation =
    std::make_shared<vision::RotateOperation>(FixRotationAngle::k0Degree);
  rotate_operation.get()->setAngle(angle_id);
  *operation = rotate_operation;
#endif
  return Status::OK();
}

void RotateOperation::setAngle(uint64_t angle_id) {
  std::dynamic_pointer_cast<RotateOp>(rotate_op_)->setAngle(angle_id);
}

MapTargetDevice RotateOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Rotate: Invalid device target. It's not CPU or Ascend.";
  }
  return MapTargetDevice::kInvalid;
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
