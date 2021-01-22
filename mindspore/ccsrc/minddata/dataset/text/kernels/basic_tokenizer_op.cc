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
#include "minddata/dataset/text/kernels/basic_tokenizer_op.h"
#include <memory>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "unicode/errorcode.h"
#include "unicode/normalizer2.h"
#include "unicode/utypes.h"

namespace mindspore {
namespace dataset {

const bool BasicTokenizerOp::kDefLowerCase = false;
const bool BasicTokenizerOp::kDefKeepWhitespace = false;
const NormalizeForm BasicTokenizerOp::kDefNormalizationForm = NormalizeForm::kNone;
const bool BasicTokenizerOp::kDefPreserveUnusedToken = true;
const bool BasicTokenizerOp::kDefWithOffsets = false;
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
const char BasicTokenizerOp::kUnusedPattern[] = "\\[CLS\\]|\\[SEP\\]|\\[UNK\\]|\\[PAD\\]|\\[MASK\\]|\\[unused\\d+\\]|";
const std::unordered_set<std::string> BasicTokenizerOp::kUnusedWords{"[CLS]", "[SEP]", "[UNK]", "[PAD]", "[MASK]"};

BasicTokenizerOp::BasicTokenizerOp(const bool &lower_case, const bool &keep_whitespace,
                                   const NormalizeForm &normalization_form, const bool &preserve_unused_token,
                                   const bool &with_offsets)
    : lower_case_(lower_case),
      keep_whitespace_(keep_whitespace),
      preserve_unused_token_(preserve_unused_token),
      with_offsets_(with_offsets),
      case_fold_(std::make_unique<CaseFoldOp>()),
      nfd_normalize_(std::make_unique<NormalizeUTF8Op>(NormalizeForm::kNfd)),
      normalization_form_(normalization_form),
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
  regex_tokenizer_ = std::make_unique<RegexTokenizerOp>(delim_pattern, keep_delim_pattern, with_offsets_);
}

Status BasicTokenizerOp::CaseFoldWithoutUnusedWords(const std::string_view &text,
                                                    const std::unordered_set<std::string> &unused_words,
                                                    std::string *output) {
  icu::ErrorCode error;
  const icu::Normalizer2 *nfkc_case_fold = icu::Normalizer2::getNFKCCasefoldInstance(error);
  CHECK_FAIL_RETURN_UNEXPECTED(error.isSuccess(), "BasicTokenizer: getNFKCCasefoldInstance failed.");
  output->clear();

  // 1. get start and end offsets of not case fold strs
  std::queue<std::pair<int, int>> offsets;  // offsets of not used words
  int start = -1;
  int len = 0;
  for (int i = 0; i < text.length(); i++) {
    if (text[i] == '[') {
      start = i;
      ++len;
    } else if (text[i] == ']' && start >= 0) {
      ++len;
      std::string word(text.substr(start, len));
      if (unused_words.find(word) != unused_words.end()) {
        offsets.push(std::make_pair(start, start + len - 1));
      }
      start = -1;
      len = 0;
    } else if (start >= 0) {
      ++len;
    }
  }

  // 2. Do not apply case fold on `unused_words`
  start = 0;
  for (int i = 0; i < text.length();) {
    std::string_view process_text;
    std::string preserve_token;
    if (offsets.empty()) {
      i = text.length();
      process_text = text.substr(start, i - start);
    } else {
      preserve_token = text.substr(offsets.front().first, offsets.front().second - offsets.front().first + 1);
      process_text = text.substr(start, offsets.front().first - start);
      i = offsets.front().second + 1;
      offsets.pop();
    }
    std::string temp;
    icu::StringByteSink<std::string> sink(&temp);
    nfkc_case_fold->normalizeUTF8(0, icu::StringPiece(process_text.data(), process_text.size()), sink, nullptr, error);
    *output += temp + preserve_token;
  }
  return Status::OK();
}

Status BasicTokenizerOp::CaseFoldWithoutUnusedWords(const std::shared_ptr<Tensor> &input,
                                                    std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input->type() == DataType::DE_STRING, "BasicTokenizer: input is not string datatype.");
  std::vector<std::string> strs(input->Size());
  int i = 0;
  for (auto iter = input->begin<std::string_view>(); iter != input->end<std::string_view>(); iter++) {
    RETURN_IF_NOT_OK(CaseFoldWithoutUnusedWords(*iter, kUnusedWords, &strs[i++]));
  }
  return Tensor::CreateFromVector(strs, input->shape(), output);
}

Status BasicTokenizerOp::Compute(const TensorRow &input, TensorRow *output) {
  IO_CHECK_VECTOR(input, output);
  CHECK_FAIL_RETURN_UNEXPECTED(input.size() == 1, "BasicTokenizer: input only support one column data.");
  if (input[0]->Rank() != 0 || input[0]->type() != DataType::DE_STRING) {
    RETURN_STATUS_UNEXPECTED("BasicTokenizer: the input should be scalar with string datatype");
  }
  std::shared_ptr<Tensor> cur_input;
  std::shared_ptr<Tensor> processed_tensor;
  if (lower_case_) {
    if (!preserve_unused_token_) {
      // to lower case
      RETURN_IF_NOT_OK(case_fold_->Compute(input[0], &processed_tensor));
    } else {
      // to lower case except words in kUnusedWords
      RETURN_IF_NOT_OK(CaseFoldWithoutUnusedWords(input[0], &processed_tensor));
    }
    cur_input = processed_tensor;
    // strip accent characters
    RETURN_IF_NOT_OK(nfd_normalize_->Compute(cur_input, &processed_tensor));
    cur_input = processed_tensor;
    RETURN_IF_NOT_OK(replace_accent_chars_->Compute(cur_input, &processed_tensor));
  } else {
    RETURN_IF_NOT_OK(common_normalize_->Compute(input[0], &processed_tensor));
  }
  // strip control characters
  cur_input = processed_tensor;
  RETURN_IF_NOT_OK(replace_control_chars_->Compute(cur_input, &processed_tensor));
  return regex_tokenizer_->Compute(TensorRow(0, {std::move(processed_tensor)}), output);
}
}  // namespace dataset
}  // namespace mindspore
