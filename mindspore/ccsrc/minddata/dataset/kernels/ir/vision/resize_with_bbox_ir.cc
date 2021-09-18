/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "minddata/dataset/kernels/ir/vision/resize_with_bbox_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/resize_with_bbox_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// ResizeWithBBoxOperation
ResizeWithBBoxOperation::ResizeWithBBoxOperation(const std::vector<int32_t> &size, InterpolationMode interpolation)
    : size_(size), interpolation_(interpolation) {}

ResizeWithBBoxOperation::~ResizeWithBBoxOperation() = default;

std::string ResizeWithBBoxOperation::Name() const { return kResizeWithBBoxOperation; }

Status ResizeWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("ResizeWithBBox", size_));
  // interpolation
  if (interpolation_ != InterpolationMode::kLinear && interpolation_ != InterpolationMode::kNearestNeighbour &&
      interpolation_ != InterpolationMode::kCubic && interpolation_ != InterpolationMode::kArea &&
      interpolation_ != InterpolationMode::kCubicPil) {
    std::string err_msg = "ResizeWithBBox: Invalid InterpolationMode, check input value of enum.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ResizeWithBBoxOperation::Build() {
  constexpr size_t dimension_zero = 0;
  constexpr size_t dimension_one = 1;
  constexpr size_t size_two = 2;

  int32_t height = size_[dimension_zero];
  int32_t width = 0;

  // User specified the width value.
  if (size_.size() == size_two) {
    width = size_[dimension_one];
  }

  return std::make_shared<ResizeWithBBoxOp>(height, width, interpolation_);
}

Status ResizeWithBBoxOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["size"] = size_;
  args["interpolation"] = interpolation_;
  *out_json = args;
  return Status::OK();
}

Status ResizeWithBBoxOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("size") != op_params.end(), "Failed to find size");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("interpolation") != op_params.end(), "Failed to find interpolation");
  std::vector<int32_t> size = op_params["size"];
  InterpolationMode interpolation = static_cast<InterpolationMode>(op_params["interpolation"]);
  *operation = std::make_shared<vision::ResizeWithBBoxOperation>(size, interpolation);
  return Status::OK();
}

#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
