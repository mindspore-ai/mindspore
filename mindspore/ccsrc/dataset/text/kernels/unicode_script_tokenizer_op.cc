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
#include "dataset/text/kernels/unicode_script_tokenizer_op.h"
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

const bool UnicodeScriptTokenizerOp::kDefKeepWhitespace = false;

Status UnicodeScriptTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
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

  UScriptCode last_script = USCRIPT_INVALID_CODE;
  icu::ErrorCode status;
  int start = 0;
  int len = 0;
  std::vector<std::string> splits;

  bool was_space = false;
  for (size_t i = 0; i < runes.size(); i++) {
    bool is_space = u_isUWhiteSpace(runes[i].rune);
    UScriptCode script = uscript_getScript(runes[i].rune, status);
    if (status.isFailure()) {
      status.reset();
      script = USCRIPT_INVALID_CODE;
    }
    // 1) Seperate UTF-8 strings of different UScriptCode values
    //    (such as: "Chinese中国" should be splited to ["Chinese", "中国"])
    // 2) Seperate whitespace and non-whitespace UTF-8 strings
    //    (such as: " ." should be split to [" ", "."])
    if (len > 0 && (script != last_script || is_space != was_space)) {
      // 3) If keep_whitespace_ is false, all the whitespace characters will be discard
      if (keep_whitespace_ || !was_space) {
        std::string temp(str.substr(start, len));
        splits.emplace_back(std::move(temp));
      }
      start = runes[i].offset;
      len = runes[i].len;
    } else {
      len += runes[i].len;
    }
    last_script = script;
    was_space = is_space;
  }

  if (len > 0 && (keep_whitespace_ || !was_space)) {
    std::string temp(str.substr(start, len));
    splits.emplace_back(std::move(temp));
  }
  // 4) If the input is empty scalar string, the output will be 1-D empty string.
  if (splits.empty()) {
    splits.emplace_back("");
  }
  *output = std::make_shared<Tensor>(splits, TensorShape({(dsize_t)splits.size()}));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
