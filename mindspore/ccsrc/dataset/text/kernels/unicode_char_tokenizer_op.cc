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
#include "dataset/text/kernels/unicode_char_tokenizer_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "cppjieba/Unicode.hpp"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;

namespace mindspore {
namespace dataset {

Status UnicodeCharTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("The input tensor should be scalar string tensor");
  }
  std::string_view str;
  RETURN_IF_NOT_OK(input->GetItemAt(&str, {}));

  RuneStrArray runes;
  if (!DecodeRunesInString(str.data(), str.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("Decode utf8 string failed.");
  }
  std::vector<std::string> splits(runes.size());
  for (size_t i = 0; i < runes.size(); i++) {
    splits[i] = str.substr(runes[i].offset, runes[i].len);
  }
  if (splits.empty()) {
    splits.emplace_back("");
  }
  *output = std::make_shared<Tensor>(splits, TensorShape({(dsize_t)splits.size()}));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
