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
#include "minddata/dataset/kernels/ir/vision/center_crop_ir.h"

#include "minddata/dataset/kernels/image/center_crop_op.h"

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
CenterCropOperation::CenterCropOperation(const std::vector<int32_t> &size) : size_(size) {}

CenterCropOperation::~CenterCropOperation() = default;

std::string CenterCropOperation::Name() const { return kCenterCropOperation; }

Status CenterCropOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("CenterCrop", size_));
  return Status::OK();
}

std::shared_ptr<TensorOp> CenterCropOperation::Build() {
  int32_t crop_height = size_[0];
  int32_t crop_width = size_[0];

  // User has specified crop_width.
  constexpr size_t size_two = 2;
  if (size_.size() == size_two) {
    crop_width = size_[1];
  }

  std::shared_ptr<CenterCropOp> tensor_op = std::make_shared<CenterCropOp>(crop_height, crop_width);
  return tensor_op;
}

Status CenterCropOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

Status CenterCropOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("size") != op_params.end(), "Failed to find size");
  std::vector<int32_t> size = op_params["size"];
  *operation = std::make_shared<CenterCropOperation>(size);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
