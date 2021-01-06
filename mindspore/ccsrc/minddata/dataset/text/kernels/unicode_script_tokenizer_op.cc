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
#include "minddata/dataset/text/kernels/unicode_script_tokenizer_op.h"
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
const bool UnicodeScriptTokenizerOp::kDefWithOffsets = false;

Status UnicodeScriptTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "UnicodeScriptTokenizer: input should be one column data.");
  if (input[0]->Rank() != 0 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED(
      "UnicodeScriptTokenizer: "
      "the input shape should be scalar and the input datatype should be string.");
  }
  std::string_view str;
  RETURN_IF_NOT_OK(input[0]->GetItemAt(&str, {}));
  RuneStrArray runes;
  if (!DecodeRunesInString(str.data(), str.size(), runes)) {
    RETURN_STATUS_UNEXPECTED("UnicodeScriptTokenizer: Decode utf8 string failed.");
  }

  std::shared_ptr<Tensor> token_tensor, offsets_start_tensor, offsets_limit_tensor;
  UScriptCode last_script = USCRIPT_INVALID_CODE;
  icu::ErrorCode status;
  int start = 0;
  int len = 0;
  std::vector<std::string> splits;
  std::vector<uint32_t> offsets_start, offsets_limit;

  bool was_space = false;
  for (size_t i = 0; i < runes.size(); i++) {
    bool is_space = u_isUWhiteSpace(runes[i].rune);
    UScriptCode script = uscript_getScript(runes[i].rune, status);
    if (status.isFailure()) {
      status.reset();
      script = USCRIPT_INVALID_CODE;
    }
    // 1) Separate UTF-8 strings of different UScriptCode values
    //    (such as: "Chinese中国" should be splited to ["Chinese", "中国"])
    // 2) Separate whitespace and non-whitespace UTF-8 strings
    //    (such as: " ." should be split to [" ", "."])
    if (len > 0 && (script != last_script || is_space != was_space)) {
      // 3) If keep_whitespace_ is false, all the whitespace characters will be discard
      if (keep_whitespace_ || !was_space) {
        offsets_start.push_back(static_cast<uint32_t>(start));
        offsets_limit.push_back(static_cast<uint32_t>(start + len));
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
    offsets_start.push_back(static_cast<uint32_t>(start));
    offsets_limit.push_back(static_cast<uint32_t>(start + len));
    std::string temp(str.substr(start, len));
    splits.emplace_back(std::move(temp));
  }
  // 4) If the input is empty scalar string, the output will be 1-D empty string.
  if (splits.empty()) {
    splits.emplace_back("");
    offsets_start.push_back(0);
    offsets_limit.push_back(0);
  }
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(splits, &token_tensor));
  output->push_back(token_tensor);
  if (with_offsets_) {
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_start, &offsets_start_tensor));
    RETURN_IF_NOT_OK(Tensor::CreateFromVector(offsets_limit, &offsets_limit_tensor));

    output->push_back(offsets_start_tensor);
    output->push_back(offsets_limit_tensor);
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
