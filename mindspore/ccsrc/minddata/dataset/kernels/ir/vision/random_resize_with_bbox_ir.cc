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

#include "minddata/dataset/kernels/ir/vision/random_resize_with_bbox_ir.h"

#ifndef ENABLE_ANDROID
#include "minddata/dataset/kernels/image/random_resize_with_bbox_op.h"
#endif

#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
#ifndef ENABLE_ANDROID
// RandomResizeWithBBoxOperation
RandomResizeWithBBoxOperation::RandomResizeWithBBoxOperation(const std::vector<int32_t> &size)
    : TensorOperation(true), size_(size) {}

RandomResizeWithBBoxOperation::~RandomResizeWithBBoxOperation() = default;

std::string RandomResizeWithBBoxOperation::Name() const { return kRandomResizeWithBBoxOperation; }

Status RandomResizeWithBBoxOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateVectorSize("RandomResizeWithBBox", size_));
  return Status::OK();
}

std::shared_ptr<TensorOp> RandomResizeWithBBoxOperation::Build() {
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

  std::shared_ptr<RandomResizeWithBBoxOp> tensor_op = std::make_shared<RandomResizeWithBBoxOp>(height, width);
  return tensor_op;
}

Status RandomResizeWithBBoxOperation::to_json(nlohmann::json *out_json) {
  (*out_json)["size"] = size_;
  return Status::OK();
}

Status RandomResizeWithBBoxOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("size") != op_params.end(), "Failed to find size");
  std::vector<int32_t> size = op_params["size"];
  *operation = std::make_shared<vision::RandomResizeWithBBoxOperation>(size);
  return Status::OK();
}
#endif
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
