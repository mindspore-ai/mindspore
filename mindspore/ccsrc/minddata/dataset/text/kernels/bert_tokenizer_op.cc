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
#include "minddata/dataset/text/kernels/bert_tokenizer_op.h"
namespace mindspore {
namespace dataset {
Status BertTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  TensorRow basic_tensor;
  RETURN_IF_NOT_OK(basic_tokenizer_.Compute(input, &basic_tensor));
  RETURN_IF_NOT_OK(wordpiece_tokenizer_.Compute(basic_tensor, output));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
