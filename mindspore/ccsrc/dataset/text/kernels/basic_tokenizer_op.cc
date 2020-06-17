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
#include "dataset/text/kernels/basic_tokenizer_op.h"
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mindspore {
namespace dataset {
const bool BasicTokenizerOp::kDefLowerCase = false;
const bool BasicTokenizerOp::kDefKeepWhitespace = false;
const NormalizeForm BasicTokenizerOp::kDefNormalizationForm = NormalizeForm::kNone;
const bool BasicTokenizerOp::kDefPreserveUnusedToken = true;
const char BasicTokenizerOp::kCommonPattern[] =
  "[!-/]"
  "|[:-@]"
  "|[\\[-`]"
  "|[{-~]"
  "|[\\p{P}]"
  "|[\\x{4E00}-\\x{9FFF}]"
  "|[\\x{3400}-\\x{4DBF}]"
  "|[\\x{20000}-\\x{2A6DF}]"
  "|[\\x{2A700}-\\x{2B73F}]"
  "|[\\x{2B740}-\\x{2B81F}]"
  "|[\\x{2B820}-\\x{2CEAF}]"
  "|[\\x{F900}-\\x{FAFF}]"
  "|[\\x{2F800}-\\x{2FA1F}]";
const char BasicTokenizerOp::kUnusedPattern[] = "\\[CLS\\]|\\[SEP\\]|\\[UNK\\]|\\[PAD\\]|\\[MASK\\]|";

BasicTokenizerOp::BasicTokenizerOp(bool lower_case, bool keep_whitespace, NormalizeForm normalization_form,
                                   bool preserve_unused_token)
    : lower_case_(lower_case),
      keep_whitespace_(keep_whitespace),
      preserve_unused_token_(preserve_unused_token),
      case_fold_(std::make_unique<CaseFoldOp>()),
      nfd_normalize_(std::make_unique<NormalizeUTF8Op>(NormalizeForm::kNfd)),
      common_normalize_(std::make_unique<NormalizeUTF8Op>(normalization_form)),
      replace_accent_chars_(std::make_unique<RegexReplaceOp>("\\p{Mn}", "")),
      replace_control_chars_(std::make_unique<RegexReplaceOp>("\\p{Cc}|\\p{Cf}", " ")) {
  std::string delim_pattern = std::string("\\s+|") + kCommonPattern;
  std::string keep_delim_pattern;
  if (keep_whitespace_) {
    keep_delim_pattern = delim_pattern;
  } else {
    keep_delim_pattern = kCommonPattern;
  }
  if (preserve_unused_token_) {
    keep_delim_pattern = kUnusedPattern + keep_delim_pattern;
    delim_pattern = kUnusedPattern + delim_pattern;
  }
  regex_tokenizer_ = std::make_unique<RegexTokenizerOp>(delim_pattern, keep_delim_pattern);
}

Status BasicTokenizerOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  if (input->Rank() != 0 || input->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("The input tensor should be scalar string tensor");
  }
  std::shared_ptr<Tensor> cur_input;
  std::shared_ptr<Tensor> processed_tensor;
  if (lower_case_) {
    // to lower case
    RETURN_IF_NOT_OK(case_fold_->Compute(input, &processed_tensor));
    cur_input = processed_tensor;
    // strip accent characters
    RETURN_IF_NOT_OK(nfd_normalize_->Compute(cur_input, &processed_tensor));
    cur_input = processed_tensor;
    RETURN_IF_NOT_OK(replace_accent_chars_->Compute(cur_input, &processed_tensor));
  } else {
    RETURN_IF_NOT_OK(common_normalize_->Compute(input, &processed_tensor));
  }
  // strip control characters
  cur_input = processed_tensor;
  RETURN_IF_NOT_OK(replace_control_chars_->Compute(cur_input, &processed_tensor));
  return regex_tokenizer_->Compute(processed_tensor, output);
}
}  // namespace dataset
}  // namespace mindspore
