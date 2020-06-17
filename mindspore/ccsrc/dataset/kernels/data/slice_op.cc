/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "dataset/kernels/data/slice_op.h"

#include "dataset/core/tensor.h"
#include "dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {
Status SliceOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Rank() == 1, "SliceOp supports 1D Tensors only for now.");

  // if `all` flag is true, output is just the input.
  if (all_) {
    *output = input;
    return Status::OK();
  }

  // if slice object was provided, indices should be empty. Generate indices from the slice object.
  if (slice_.valid() && indices_.empty()) {
    dsize_t len = input->shape()[0];
    indices_ = slice_.Indices(len);
    return input->Slice(output, indices_);
  }

  // if indices are not empty, slices should be invalid, use indices_ to slice
  if (!indices_.empty() && !slice_.valid()) {
    return input->Slice(output, indices_);
  }
  RETURN_STATUS_UNEXPECTED("The indexing parameters are invalid");
}
}  // namespace dataset
}  // namespace mindspore
