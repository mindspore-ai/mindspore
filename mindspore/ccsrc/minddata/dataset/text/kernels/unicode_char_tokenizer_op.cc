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

#include "minddata/dataset/text/kernels/unicode_char_tokenizer_op.h"

#include <memory>
#include <string_view>
#include <utility>
#include <vector>

#include "cppjieba/Unicode.hpp"

using cppjieba::DecodeRunesInString;
using cppjieba::RuneStrArray;

namespace mindspore {
namespace dataset {

Status UnicodeCharTokenizerOp::Tokenize(std::string_view str, std::vector<std::string> *splits,
                                        std::vector<uint32_t> *offsets_start, std::vector<uint32_t> *offsets_limit) {
  RETURN_UNEXPECTED_IF_NULL(splits);
  RETURN_UNEXPECTED_IF_NULL(offsets_start);
  RETURN_UNEXPECTED_IF_NULL(offsets_limit);
  RuneStrArray runes;
  if (!DecodeRunesInString(str.data(), str.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("UnicodeCharTokenizer: Decode utf8 string failed.");
  }
  std::vector<std::string> words(runes.size());
  for (size_t i = 0; i < runes.size(); i++) {
    offsets_start->push_back(runes[i].offset);
    offsets_limit->push_back(runes[i].offset + runes[i].len);
    words[i] = str.substr(runes[i].offset, runes[i].len);
  }
  *splits = std::move(words);
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
