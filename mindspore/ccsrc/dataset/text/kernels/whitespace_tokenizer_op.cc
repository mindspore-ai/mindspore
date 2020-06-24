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
#include "dataset/text/kernels/whitespace_tokenizer_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cppjieba/Unicode.hpp"
#include "unicode/errorcode.h"
#include "unicode/uchar.h"
#include "unicode/uscript.h"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;

namespace mindspore {
namespace dataset {
Status WhitespaceTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
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
  std::vector<std::string> splits;
  int start = 0;
  int len = 0;
  for (size_t i = 0; i < runes.size(); i++) {
    if (u_isUWhiteSpace(runes[i].rune)) {
      if (len > 0) {
        std::string temp(str.substr(start, len));
        splits.emplace_back(std::move(temp));
        len = 0;
      }
    } else {
      if (len == 0) {
        start = runes[i].offset;
      }
      len += runes[i].len;
    }
  }
  if (len > 0) {
    std::string temp(str.substr(start, len));
    splits.emplace_back(std::move(temp));
  }
  if (splits.empty()) {
    splits.emplace_back("");
  }
  *output = std::make_shared<Tensor>(splits, TensorShape({(dsize_t)splits.size()}));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
