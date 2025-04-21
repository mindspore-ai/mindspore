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
#include "minddata/dataset/text/kernels/to_vectors_op.h"

namespace mindspore {
namespace dataset {
ToVectorsOp::ToVectorsOp(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init,
                         bool lower_case_backup)
    : vectors_(vectors), unk_init_(unk_init), lower_case_backup_(lower_case_backup) {}

Status ToVectorsOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "ToVectors: input tensor type should be string.");
  CHECK_FAIL_RETURN_UNEXPECTED(unk_init_.size() == 0 || unk_init_.size() == vectors_->Dim(),
                               "ToVectors: unk_init must be the same length as vectors, but got unk_init: " +
                                 std::to_string(unk_init_.size()) + " and vectors: " + std::to_string(vectors_->Dim()));

  std::vector<float> vectors_vec;
  int len = 0;
  for (auto itr = input->begin<std::string_view>(); itr != input->end<std::string_view>(); ++itr) {
    std::vector<float> vectors_value = vectors_->Lookup(std::string(*itr), unk_init_, lower_case_backup_);
    CHECK_FAIL_RETURN_UNEXPECTED(!vectors_value.empty(), "ToVectors: invalid data, token: \"" + std::string(*itr) +
                                                           "\" doesn't exist in vectors and no unk_init is specified.");
    vectors_vec.insert(vectors_vec.end(), vectors_value.begin(), vectors_value.end());
    len++;
  }

  int dim = static_cast<int>(vectors_vec.size() / len);
  if (vectors_vec.size() == dim) {
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(vectors_vec, output));
  } else {
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(vectors_vec, TensorShape({len, dim}), output));
  }
  return Status::OK();
}

Status ToVectorsOp::OutputType(const std::vector<DataType> &inputs, std::vector<DataType> &outputs) {
  CHECK_FAIL_RETURN_UNEXPECTED(inputs.size() == NumInput() && outputs.size() == NumOutput(),
                               "ToVectors: input and output size don't match.");
  CHECK_FAIL_RETURN_UNEXPECTED(inputs[0] == DataType::DE_STRING, "ToVectors: input tensor type should be string.");
  outputs[0] = DataType(DataType::DE_FLOAT32);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
