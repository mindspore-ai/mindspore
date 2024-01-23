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
#include "minddata/dataset/kernels/ir/vision/perspective_ir.h"

#include "minddata/dataset/kernels/image/perspective_op.h"
#include "minddata/dataset/util/validators.h"
#if !defined(BUILD_LITE) && defined(ENABLE_D)
#include "minddata/dataset/kernels/image/dvpp/ascend910b/dvpp_perspective.h"
#endif
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
PerspectiveOperation::PerspectiveOperation(const std::vector<std::vector<int32_t>> &start_points,
                                           const std::vector<std::vector<int32_t>> &end_points,
                                           InterpolationMode interpolation, const std::string &device_target)
    : start_points_(start_points),
      end_points_(end_points),
      interpolation_(interpolation),
      device_target_(device_target) {}

PerspectiveOperation::~PerspectiveOperation() = default;

Status PerspectiveOperation::ValidateParams() {
  // device target
  if (device_target_ != "CPU" && device_target_ != "Ascend") {
    std::string err_msg = "Perspective: Invalid device target. It's not CPU or Ascend.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "Perspective: Invalid InterpolationMode, check input value of enum.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  const size_t kListSize = 4;
  const size_t kPointSize = 2;
  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    start_points_.size() == kListSize,
    "Perspective: start_points should be in length of 4, but got: " + std::to_string(start_points_.size()));

  CHECK_FAIL_RETURN_SYNTAX_ERROR(
    end_points_.size() == kListSize,
    "Perspective: end_points should be in length of 4, but got: " + std::to_string(end_points_.size()));

  for (int i = 0; i < kListSize; i++) {
    if (start_points_[i].size() != kPointSize) {
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Perspective: each element in start_points should be in length of 2.");
    }
    if (end_points_[i].size() != kPointSize) {
      LOG_AND_RETURN_STATUS_SYNTAX_ERROR("Perspective: each element in end_points should be in length of 2.");
    }
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> PerspectiveOperation::Build() {
  if (device_target_ == "CPU") {
    std::shared_ptr<PerspectiveOp> tensor_op =
      std::make_shared<PerspectiveOp>(start_points_, end_points_, interpolation_);
    return tensor_op;
#if !defined(BUILD_LITE) && defined(ENABLE_D)
  } else if (device_target_ == "Ascend") {
    std::shared_ptr<DvppPerspectiveOp> dvpp_tensor_op =
      std::make_shared<DvppPerspectiveOp>(start_points_, end_points_, interpolation_);
    return dvpp_tensor_op;
#endif
  } else {
    MS_LOG(ERROR) << "Perspective: Invalid device target. It's not CPU or Ascend.";
    return nullptr;
  }
}  // namespace vision

Status PerspectiveOperation::to_json(nlohmann::json *out_json) {
  RETURN_UNEXPECTED_IF_NULL(out_json);
  nlohmann::json args;
  args["start_points"] = start_points_;
  args["end_points"] = end_points_;
  args["interpolation"] = interpolation_;
  args["device_target"] = device_target_;
  *out_json = args;
  return Status::OK();
}

Status PerspectiveOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_UNEXPECTED_IF_NULL(operation);
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "start_points", kPerspectiveOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "end_points", kPerspectiveOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "interpolation", kPerspectiveOperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "device_target", kPerspectiveOperation));
  std::vector<std::vector<int32_t>> start_points = op_params["start_points"];
  std::vector<std::vector<int32_t>> end_points = op_params["end_points"];
  auto interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  std::string device_target = op_params["device_target"];
  *operation = std::make_shared<vision::PerspectiveOperation>(start_points, end_points, interpolation, device_target);
  return Status::OK();
}

MapTargetDevice PerspectiveOperation::Type() {
  if (device_target_ == "CPU") {
    return MapTargetDevice::kCpu;
  } else if (device_target_ == "Ascend") {
    return MapTargetDevice::kAscend910B;
  } else {
    MS_LOG(ERROR) << "Perspective: Invalid device target. It's not CPU or Ascend.";
    return MapTargetDevice::kInvalid;
  }
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
