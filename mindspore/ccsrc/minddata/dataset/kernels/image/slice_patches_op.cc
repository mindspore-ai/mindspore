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
#include "minddata/dataset/kernels/image/slice_patches_op.h"
#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
const int32_t SlicePatchesOp::kDefNumH = 1;
const int32_t SlicePatchesOp::kDefNumW = 1;
const uint8_t SlicePatchesOp::kDefFillV = 0;
const SliceMode SlicePatchesOp::kDefSliceMode = SliceMode::kPad;

SlicePatchesOp::SlicePatchesOp(int32_t num_height, int32_t num_width, SliceMode slice_mode, uint8_t fill_value)
    : num_height_(num_height), num_width_(num_width), slice_mode_(slice_mode), fill_value_(fill_value) {}

Status SlicePatchesOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "Input tensor size should be 1.");

  auto in_tensor = input[0];
  auto in_type = in_tensor->type();
  auto in_shape = in_tensor->shape();

  CHECK_FAIL_RETURN_UNEXPECTED(in_type.IsNumeric(), "Input Tensor type should be numeric.");
  CHECK_FAIL_RETURN_UNEXPECTED(in_shape.Rank() >= 2, "Input Tensor rank should be greater than 2.");

  std::vector<std::shared_ptr<Tensor>> out;
  RETURN_IF_NOT_OK(SlicePatches(in_tensor, &out, num_height_, num_width_, slice_mode_, fill_value_));
  (void)std::copy(out.begin(), out.end(), std::back_inserter(*output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
