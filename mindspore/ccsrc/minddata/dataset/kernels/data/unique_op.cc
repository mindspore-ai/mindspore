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

#include "minddata/dataset/kernels/data/unique_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/tensor_op.h"

namespace mindspore {
namespace dataset {

Status UniqueOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "Unique: only support 1D input");

  auto in_tensor = input[0];
  auto in_tensor_shape = in_tensor->shape();
  auto in_tensor_type = in_tensor->type();

  CHECK_FAIL_RETURN_UNEXPECTED(in_tensor_type.IsNumeric(), "Unique: Tensor type must be numeric.");
  CHECK_FAIL_RETURN_UNEXPECTED(in_tensor_shape.Rank() >= 2,
                               "Unique: input must be at least 2-D in order to do unique op.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    in_tensor->Size() <= std::numeric_limits<int32_t>::max(),
    "Unique: Unique does not support input tensor large than " + std::to_string(std::numeric_limits<int32_t>::max()));

  RETURN_IF_NOT_OK(in_tensor->Reshape(TensorShape({in_tensor->Size()})));

  std::shared_ptr<Tensor> out;
  std::shared_ptr<Tensor> out_idx;
  std::shared_ptr<Tensor> out_cnt;

  RETURN_IF_NOT_OK(Unique(in_tensor, &out, &out_idx, &out_cnt));
  output->push_back(out);
  output->push_back(out_idx);
  output->push_back(out_cnt);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
