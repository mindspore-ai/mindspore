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

#include "minddata/dataset/kernels/ir/vision/resized_crop_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/resized_crop_op.h"
#endif
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_resized_crop.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// ResizedCropOperation
ResizedCropOperation::ResizedCropOperation(int32_t top, int32_t left, int32_t height, int32_t width,
                                           const std::vector<int32_t> &size, InterpolationMode interpolation,
                                           const std::string &device_target)
    : top_(top),
      left_(left),
      height_(height),
      width_(width),
      size_(size),
      interpolation_(interpolation),
      device_target_(device_target) {}

ResizedCropOperation::~ResizedCropOperation() = default;

std::string ResizedCropOperation::Name() const { return kResizedCropOperation; }

Status ResizedCropOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("ResizedCrop", "top", top_));
  RETURN_IF_NOT_OK(ValidateIntScalarNonNegative("ResizedCrop", "left", left_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("ResizedCrop", "height", height_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("ResizedCrop", "width", width_));
  RETURN_IF_NOT_OK(ValidateVectorSize("ResizedCrop", size_));

  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "ResizedCrop: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "ResizedCrop: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ResizedCropOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<ResizedCropOp> tensor_op =
      std::make_shared<ResizedCropOp>(top_, left_, height_, width_, size_, interpolation_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppResizedCropOp> dvpp_tensor_op =
      std::make_shared<DvppResizedCropOp>(top_, left_, height_, width_, size_, interpolation_);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "ResizedCrop: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status ResizedCropOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["top"] = top_;
  args["left"] = left_;
  args["height"] = height_;
  args["width"] = width_;
  args["size"] = size_;
  args["interpolation"] = interpolation_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status ResizedCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "top", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "left", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "height", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "width", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kResizedCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kResizedCropOperation));
  int32_t top = op_params["top"];
  int32_t left = op_params["left"];
  int32_t height = op_params["height"];
  int32_t width = op_params["width"];
  std::vector<int32_t> size = op_params["size"];
  auto interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  std::string device_target = op_params["device_target"];

  *operation = std::make_shared<ResizedCropOperation>(top, left, height, width, size, interpolation, device_target);
  return Status::OK();
}

MapTargetDevice ResizedCropOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "ResizedCrop: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
