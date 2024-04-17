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

#include "minddata/dataset/text/kernels/add_token_op.h"

#include "minddata/dataset/text/kernels/data_utils.h"

namespace mindspore {
namespace dataset {
Status AddTokenOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->Rank() == 1 || input->Rank() == 2,
                               "AddToken: input tensor rank should be 1 or 2.");
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "AddToken: input tensor type should be string.");

  return AddToken(input, output, token_, begin_);
}

Status AddTokenOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  outputs.clear();
  TensorShape input_shape = inputs[0];
  std::vector<dsize_t> output_shape_vector = input_shape.AsVector();
  output_shape_vector[input_shape.Size() == 1 ? 0 : 1] += 1;
  TensorShape out = TensorShape(output_shape_vector);
  (void)outputs.emplace_back(out);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
