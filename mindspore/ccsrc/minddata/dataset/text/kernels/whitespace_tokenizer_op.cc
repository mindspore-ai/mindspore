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
#include "minddata/dataset/text/kernels/whitespace_tokenizer_op.h"
#include <string_view>
#include <utility>
#include <vector>

#include "cppjieba/Unicode.hpp"
#include "unicode/uchar.h"
#include "unicode/uscript.h"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;

namespace mindspore {
namespace dataset {
Status WhitespaceTokenizerOp::Tokenize(std::string_view str, std::vector<std::string> *splits,
                                       std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) {
  RETURN_UNEXPECTED_IF_NULL(splits);
  RETURN_UNEXPECTED_IF_NULL(offsets_start);
  RETURN_UNEXPECTED_IF_NULL(offsets_limit);
  RuneStrArray runes;
  if (!DecodeRunesInString(str.data(), str.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("WhitespaceTokenizer: Decode utf8 string failed.");
  }

  int start = 0;
  int len = 0;
  for (size_t i = 0; i < runes.size(); i++) {
    if (u_isUWhiteSpace(runes[i].rune)) {
      if (len > 0) {
        offsets_start->push_back(static_cast<uint32_t>(start));
        offsets_limit->push_back(static_cast<uint32_t>(start + len));
        std::string temp(str.substr(start, len));
        (void)splits->emplace_back(std::move(temp));
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
    offsets_start->push_back(static_cast<uint32_t>(start));
    offsets_limit->push_back(static_cast<uint32_t>(start + len));
    std::string temp(str.substr(start, len));
    (void)splits->emplace_back(std::move(temp));
  }
  if (splits->empty()) {
    (void)splits->emplace_back("");
    offsets_start->push_back(0);
    offsets_limit->push_back(0);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
