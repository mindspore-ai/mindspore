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
#include "minddata/dataset/kernels/ir/vision/crop_ir.h"

#include "minddata/dataset/kernels/image/crop_op.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_crop_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
CropOperation::CropOperation(const std::vector<int32_t> &coordinates, const std::vector<int32_t> &size,
                             const std::string &device_target)
    : coordinates_(coordinates), size_(size), device_target_(device_target) {}

CropOperation::~CropOperation() = default;

std::string CropOperation::Name() const { return kCropOperation; }

Status CropOperation::ValidateParams() {
  // We have to limit crop size due to library restrictions, optimized to only iterate over size_ once
  // we don't check the coordinates here because we don't have access to image dimensions
  RETURN_IF_NOT_OK(ValidateVectorSize("Crop", size_));

  constexpr size_t kSizeSize = 2;
  if (coordinates_.size() != kSizeSize) {
    std::string err_msg = "Crop: 'coordinates' must be a vector of two values.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  RETURN_IF_NOT_OK(ValidateVectorNonNegative("Crop", "coordinates", coordinates_));
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "ResizedCrop: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> CropOperation::Build() {
  int32_t y, x, height, width;

  x = coordinates_[0];
  y = coordinates_[1];

  height = size_[0];
  width = size_[0];
  // User has specified crop_width.
  constexpr size_t size_two = 2;
  if (size_.size() == size_two) {
    width = size_[1];
  }

  if (device_target_ == "CPU") {
    std::shared_ptr<CropOp> tensor_op = std::make_shared<CropOp>(y, x, height, width);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppCropOp> dvpp_tensor_op = std::make_shared<DvppCropOp>(y, x, height, width);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "ResizedCrop: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

Status CropOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  (*out_json)["coordinates"] = coordinates_;
  (*out_json)["size"] = size_;
  (*out_json)["device_target"] = device_target_;
  return Status::OK();
}

Status CropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "coordinates", kCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kCropOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kCropOperation));
  std::vector<int32_t> coordinates = op_params["coordinates"];
  std::vector<int32_t> size = op_params["size"];
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<CropOperation>(coordinates, size, device_target);
  return Status::OK();
}

MapTargetDevice CropOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Crop: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
