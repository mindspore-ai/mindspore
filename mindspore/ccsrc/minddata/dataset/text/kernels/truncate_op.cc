/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/text/kernels/truncate_op.h"

#include "minddata/dataset/core/tensor.h"
#include "minddata/dataset/kernels/data/slice_op.h"
#include "minddata/dataset/kernels/tensor_op.h"
#include "minddata/dataset/text/kernels/data_utils.h"

namespace mindspore {
namespace dataset {
Status TruncateOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  constexpr int kMaxSeqRank = 2;
  CHECK_FAIL_RETURN_UNEXPECTED(input->shape().Rank() == 1 || input->shape().Rank() == kMaxSeqRank,
                               "Truncate: the input tensor should be of dimension 1 or 2.");
  CHECK_FAIL_RETURN_UNEXPECTED(
    input->type() == DataType::DE_STRING || input->type().IsNumeric(),
    "Truncate: Truncate: the input tensor should be in type of [bool, int, float, double, string].");
  return Truncate(input, output, max_seq_len_);
}

Status TruncateOp::OutputShape(const std::vector<TensorShape> &inputs, std::vector<TensorShape> &outputs) {
  RETURN_IF_NOT_OK(TensorOp::OutputShape(inputs, outputs));
  constexpr int kMaxSeqRank = 2;
  CHECK_FAIL_RETURN_UNEXPECTED(inputs[0].Rank() == 1 || inputs[0].Rank() == kMaxSeqRank,
                               "Truncate: the input tensor should be of dimension 1 or 2.");
  if (inputs[0].Rank() == 1) {
    outputs.clear();
    auto shape = inputs[0].AsVector();
    int length = shape[0];
    shape[0] = std::min(length, max_seq_len_);
    (void)outputs.emplace_back(TensorShape{shape});
  } else {
    outputs.clear();
    auto shape = inputs[0].AsVector();
    int length = shape[1];
    shape[1] = std::min(length, max_seq_len_);
    (void)outputs.emplace_back(TensorShape{shape});
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
