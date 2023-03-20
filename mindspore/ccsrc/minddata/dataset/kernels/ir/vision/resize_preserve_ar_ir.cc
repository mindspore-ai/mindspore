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
#include "minddata/dataset/kernels/ir/vision/resize_preserve_ar_ir.h"

#include "minddata/dataset/kernels/image/resize_preserve_ar_op.h"

#include "minddata/dataset/kernels/ir/validators.h"
#include "minddata/dataset/util/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// ResizePreserveAROperation
ResizePreserveAROperation::ResizePreserveAROperation(int32_t height, int32_t width, int32_t img_orientation)
    : height_(height), width_(width), img_orientation_(img_orientation) {}

ResizePreserveAROperation::~ResizePreserveAROperation() = default;

std::string ResizePreserveAROperation::Name() const { return kResizePreserveAROperation; }

Status ResizePreserveAROperation::ValidateParams() {
  constexpr int64_t max_img_orientation = 8;
  if (img_orientation_ < 1 || img_orientation_ > max_img_orientation) {
    std::string err_msg =
      "ResizePreserveAR: img_orientation must be in range of [1, 8], got: " + std::to_string(img_orientation_);
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> ResizePreserveAROperation::Build() {
  return std::make_shared<ResizePreserveAROp>(height_, width_, img_orientation_);
}

Status ResizePreserveAROperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["height"] = height_;
  args["width"] = width_;
  args["img_orientation"] = img_orientation_;
  *out_json = args;
  return Status::OK();
}

Status ResizePreserveAROperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "height", kResizePreserveAROperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "width", kResizePreserveAROperation));
  RETURN_IF_NOT_OK(ValidateParamInJson(op_params, "img_orientation", kResizePreserveAROperation));
  int32_t height = op_params["height"];
  int32_t width = op_params["width"];
  int32_t img_orientation = op_params["img_orientation"];
  *operation = std::make_shared<vision::ResizePreserveAROperation>(height, width, img_orientation);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
