/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/kernels/ir/vision/slice_patches_ir.h"
#include "minddata/dataset/kernels/image/slice_patches_op.h"
#include "minddata/dataset/kernels/ir/validators.h"

namespace mindspore {
namespace dataset {
namespace vision {
// SlicePatchesOperation
SlicePatchesOperation::SlicePatchesOperation(int32_t num_height, int32_t num_width, SliceMode slice_mode,
                                             uint8_t fill_value)
    : TensorOperation(),
      num_height_(num_height),
      num_width_(num_width),
      slice_mode_(slice_mode),
      fill_value_(fill_value) {}

SlicePatchesOperation::~SlicePatchesOperation() = default;

std::string SlicePatchesOperation::Name() const { return kSlicePatchesOperation; }

Status SlicePatchesOperation::ValidateParams() {
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("SlicePatches", "num_height", num_height_));
  RETURN_IF_NOT_OK(ValidateIntScalarPositive("SlicePatches", "num_width", num_width_));
  return Status::OK();
}

std::shared_ptr<TensorOp> SlicePatchesOperation::Build() {
  auto tensor_op = std::make_shared<SlicePatchesOp>(num_height_, num_width_, slice_mode_, fill_value_);
  return tensor_op;
}

Status SlicePatchesOperation::to_json(nlohmann::json *out_json) {
  nlohmann::json args;
  args["num_height"] = num_height_;
  args["num_width"] = num_width_;
  args["slice_mode"] = slice_mode_;
  args["fill_value"] = fill_value_;
  *out_json = args;
  return Status::OK();
}

Status SlicePatchesOperation::from_json(nlohmann::json op_params, std::shared_ptr<TensorOperation> *operation) {
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("num_height") != op_params.end(), "Failed to find num_height");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("num_width") != op_params.end(), "Failed to find num_width");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("slice_mode") != op_params.end(), "Failed to find slice_mode");
  CHECK_FAIL_RETURN_UNEXPECTED(op_params.find("fill_value") != op_params.end(), "Failed to find fill_value");
  int32_t num_height = op_params["num_height"];
  int32_t num_width = op_params["num_width"];
  SliceMode slice_mode = static_cast<SliceMode>(op_params["slice_mode"]);
  uint8_t fill_value = op_params["fill_value"];
  *operation = std::make_shared<vision::SlicePatchesOperation>(num_height, num_width, slice_mode, fill_value);
  return Status::OK();
}
}  // namespace vision
}  // namespace dataset
}  // namespace mindspore
