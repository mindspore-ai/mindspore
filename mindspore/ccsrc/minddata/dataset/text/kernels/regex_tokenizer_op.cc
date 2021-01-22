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
#include "minddata/dataset/text/kernels/regex_tokenizer_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mindspore {
namespace dataset {

const bool RegexTokenizerOp::kDefWithOffsets = false;

Status RegexTokenizerOp::GetUnicodeSubstr(const icu::UnicodeString &input, const int &start, const int &len,
                                          std::string *out_utf8, icu::UnicodeString *out_unicode) const {
  CHECK_FAIL_RETURN_UNEXPECTED((out_utf8 != nullptr || out_unicode != nullptr), "RegexTokenizer: get token failed.");
  int total_len = input.length();
  int end = start + len;
  CHECK_FAIL_RETURN_UNEXPECTED((start >= 0 && len > 0 && end <= total_len),
                               "RegexTokenizer: token offsets is out of range");
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

Status RegexTokenizerOp::GetRegexTokens(const std::string &text, std::vector<std::string> *out_tokens,
                                        std::vector<uint32_t> *offsets_start,
                                        std::vector<uint32_t> *offsets_limit) const {
  UErrorCode status = U_ZERO_ERROR;
  out_tokens->clear();
  icu::RegexMatcher token_matcher(delim_pattern_, 0, status);
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status),
                               "RegexTokenizer: create ICU RegexMatcher failed, you may input one error pattern");
  icu::RegexMatcher delim_matcher(keep_delim_pattern_, 0, status);
  CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status),
                               "RegexTokenizer: create ICU RegexMatcher failed, you may input one error pattern");

  icu::UnicodeString utext(icu::UnicodeString::fromUTF8(text));
  token_matcher.reset(utext);

  int text_start_index = 0;
  int token_start_index = 0;
  status = U_ZERO_ERROR;
  while (token_matcher.find(status) && U_SUCCESS(status)) {
    int deli_start_index = token_matcher.start(status);
    CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "RegexTokenizer: get RegexMatcher matched start index failed");
    int deli_end_index = token_matcher.end(status);
    CHECK_FAIL_RETURN_UNEXPECTED(U_SUCCESS(status), "RegexTokenizer: get RegexMatcher matched start index failed");

    // Add non-empty token
    int token_len = deli_start_index - token_start_index;
    if (token_len > 0) {
      std::string token;
      uint32_t token_offset = 0;
      RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, token_start_index, token_len, &token));
      token_offset = token.length();
      out_tokens->emplace_back(std::move(token));
      offsets_start->push_back(static_cast<uint32_t>(text_start_index));
      offsets_limit->push_back(static_cast<uint32_t>(text_start_index + token_offset));
      text_start_index += token_offset;
    }

    int delim_len = deli_end_index - deli_start_index;
    if (delim_len > 0) {
      icu::UnicodeString delim_str;
      std::string delim_utf8_str;
      uint32_t delim_str_offset = 0;
      RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, deli_start_index, delim_len, &delim_utf8_str, &delim_str));
      delim_matcher.reset(delim_str);
      delim_str_offset = delim_utf8_str.length();
      if (keep_delim_ && delim_matcher.matches(status) && U_SUCCESS(status)) {
        out_tokens->emplace_back(std::move(delim_utf8_str));
        offsets_start->push_back(static_cast<uint32_t>(text_start_index));
        offsets_limit->push_back(static_cast<uint32_t>(text_start_index + delim_str_offset));
      }
      text_start_index += delim_str_offset;
    }
    token_start_index = deli_end_index;
  }

  if (token_start_index < utext.length()) {
    std::string temp;
    uint32_t temp_offset = 0;
    RETURN_IF_NOT_OK(GetUnicodeSubstr(utext, token_start_index, utext.length() - token_start_index, &temp));
    temp_offset = temp.length();
    out_tokens->emplace_back(std::move(temp));
    offsets_start->push_back(static_cast<uint32_t>(text_start_index));
    offsets_limit->push_back(static_cast<uint32_t>(text_start_index + temp_offset));
  }
  return Status::OK();
}

Status RegexTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "RegexTokenizer: input should be one column data");
  if (input[0]->Rank() != 0 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED(
      "RegexTokenizer: the input shape should be scalar and "
      "the input datatype should be string.");
  }
  std::string_view text;
  std::vector<std::string> tokens;
  std::vector<uint32_t> offsets_start;
  std::vector<uint32_t> offsets_limit;
  std::shared_ptr<Tensor> token_tensor, offsets_start_tensor, offsets_limit_tensor;
  RETURN_IF_NOT_OK(input[0]->GetItemAt(&text, {}));
  RETURN_IF_NOT_OK(GetRegexTokens(std::string(text.data(), text.size()), &tokens, &offsets_start, &offsets_limit));
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(std::move(tokens), &token_tensor));
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
