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
#include "minddata/dataset/kernels/ir/vision/resize_ir.h"

#include "minddata/dataset/kernels/image/resize_op.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_resize_op.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// ResizeOperation
ResizeOperation::ResizeOperation(const std::vector<int32_t> &size, InterpolationMode interpolation,
                                 const std::string &device_target)
    : size_(size), interpolation_(interpolation), device_target_(device_target) {}

ResizeOperation::~ResizeOperation() = default;

std::string ResizeOperation::Name() const { return kResizeOperation; }

Status ResizeOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("Resize", size_));
  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "Resize: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Resize: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ResizeOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  // If size is a single value, the smaller edge of the image will be
  // resized to this value with the same image aspect ratio.
  int32_t height = size_[dimension_zero];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }

  if (device_target_ == "CPU") {
    return std::make_shared<ResizeOp>(height, width, interpolation_);
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    return std::make_shared<DvppResizeOp>(height, width, interpolation_);
#endif
  } else {
    MS_LOG(ERROR) << "Resize: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}

MapTargetDevice ResizeOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Resize: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}

Status ResizeOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["size"] = size_;
  args["interpolation"] = interpolation_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status ResizeOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "size", kResizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kResizeOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kResizeOperation));
  std::vector<int32_t> size = op_params["size"];
  auto interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::ResizeOperation>(size, interpolation, device_target);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
