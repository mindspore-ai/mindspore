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
#include "minddata/dataset/text/kernels/tokenizer_op.h"
#include "minddata/dataset/text/kernels/data_utils.h"

namespace mindspore {
namespace dataset {
const bool TokenizerOp::kDefWithOffsets = false;

Status TokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, Name() + ": input should be one column data.");
  if (input[0]->Rank() != 0 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED(Name() + ": the input shape should be scalar and the input datatype should be string.");
  }
  std::string_view str;
  RETURN_IF_NOT_OK(input[0]->GetItemAt(&str, {}));
  std::shared_ptr<Tensor> token_tensor;
  std::vector<uint32_t> offsets_start, offsets_limit;
  std::vector<std::string> splits;
  RETURN_IF_NOT_OK(Tokenize(str, &splits, &offsets_start, &offsets_limit));

  if (splits.empty()) {
    (void)splits.emplace_back("");
    offsets_start.push_back(0);
    offsets_limit.push_back(0);
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(splits, &token_tensor));
  output->push_back(token_tensor);
  if (with_offsets_) {
    RETURN_IF_NOT_OK(AppendOffsetsHelper(offsets_start, offsets_limit, output));
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
