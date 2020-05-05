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
#include "dataset/text/kernels/regex_tokenizer_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mindspore {
namespace dataset {
Status RegexTokenizerOp::GetUnicodeSubstr(const icu::UnicodeString &input, int start, int len, std::string *out_utf8,
                                          icu::UnicodeString *out_unicode) const {
  CHECK_FAIL_RETURN_UNEXPECTED((out_utf8 != nullptr || out_unicode != nullptr), "Wrong input");
  int total_len = input.length();
  int end = start + len;
  CHECK_FAIL_RETURN_UNEXPECTED((start >= 0 && len > 0 && end <= total_len), "Out of range");
  icu::UnicodeString temp;
  input.extract(start, len, temp);
  if (out_utf8 != nullptr) {
    temp.toUTF8String(*out_utf8);
  }
  if (out_unicode != nullptr) {
    *out_unicode = temp;
  }
  return Status::OK();
}

Status RegexTokenizerOp::GetRegexTokens(const std::string &text, std::vector<std::string> *out_tokens) const {
  UErrorCode status = U_ZERO_ERROR;
  out_tokens->clear();
  icu::RegexMatcher token_matcher(delim_pattern_, 0, status);
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "Create icu RegexMatcher failed, you may input one error pattern");
  icu::RegexMatcher delim_matcher(keep_delim_pattern_, 0, status);
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "Create icu RegexMatcher failed, you may input one error pattern");

  icu::UnicodeString utext(icu::UnicodeString::fromUTF8(text));
  token_matcher.reset(utext);

  int token_start_index = 0;
  status = U_ZERO_ERROR;
  while (token_matcher.find(status) && U_SUCCESS(status)) {
    int deli_start_index = token_matcher.start(status);
    CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "Get RegexMatcher matched start index failed");
    int deli_end_index = token_matcher.end(status);
    CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "Get RegexMatcher matched start index failed");

    // Add non-empty token
    int token_len = deli_start_index - token_start_index;
    if (token_len > 0) {
      std::string token;
      RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, token_start_index, token_len, &token));
      out_tokens->emplace_back(std::move(token));
    }

    int delim_len = deli_end_index - deli_start_index;
    if (keep_delim_ && delim_len > 0) {
      icu::UnicodeString delim_str;
      std::string delim_utf8_str;
      RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, deli_start_index, delim_len, &delim_utf8_str, &delim_str));
      delim_matcher.reset(delim_str);
      if (delim_matcher.matches(status) && U_SUCCESS(status)) {
        out_tokens->emplace_back(std::move(delim_utf8_str));
      }
    }
    token_start_index = deli_end_index;
  }

  if (token_start_index < utext.length()) {
    std::string temp;
    RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, token_start_index, utext.length() - token_start_index, &temp));
    out_tokens->emplace_back(std::move(temp));
  }
  return Status::OK();
}

Status RegexTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("The input tensor should be scalar string tensor");
  }
  std::string_view text;
  RETURN_IF_NOT_OK(input->GetItemAt(&text, {}));
  std::vector<std::string> tokens;
  RETURN_IF_NOT_OK(GetRegexTokens(std::string(text.data(), text.size()), &tokens));
  *output = std::make_shared<Tensor>(std::move(tokens), TensorShape({(dsize_t)tokens.size()}));
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
