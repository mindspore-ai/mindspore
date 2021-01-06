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
#include "minddata/dataset/kernels/data/concatenate_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/data_utils.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

Status ConcatenateOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  RETURN_IF_NOT_OK(Concatenate(input, output, axis_, prepend_, append_));
  return Status::OK();
}

Status ConcatenateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));

  std::vector<TensorShape> inputs_copy;
  inputs_copy.push_back(inputs[0].Squeeze());

  CHECK_FAIL_RETURN_UNEXPECTED(inputs.at(0).Rank() == 1, "Concatenate: only 1D input supported");

  outputs.clear();
  dsize_t output_shape = 0;
  output_shape = output_shape + inputs.at(0).NumOfElements();
  if (prepend_ != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(prepend_->shape().Rank() == 1, "Concatenate: only 1D prepend supported");
    output_shape = output_shape + prepend_->shape().NumOfElements();
  }
  if (append_ != nullptr) {
    CHECK_FAIL_RETURN_UNEXPECTED(append_->shape().Rank() == 1, "Concatenate: only 1D append supported");
    output_shape = output_shape + append_->shape().NumOfElements();
  }

  outputs.emplace_back(std::vector<dsize_t>{output_shape});
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
