/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <unistd.h>

#include "minddata/dataset/include/text.h"

namespace mindspore {
namespace dataset {

// Transform operations for text.
namespace text {

// FUNCTIONS TO CREATE TEXT OPERATIONS
// (In alphabetical order)

#ifndef _WIN32
std::shared_ptr<BasicTokenizerOperation> BasicTokenizer(bool lower_case, bool keep_whitespace,
                                                        const NormalizeForm normalize_form, bool preserve_unused_token,
                                                        bool with_offsets) {
  auto op = std::make_shared<BasicTokenizerOperation>(lower_case, keep_whitespace, normalize_form,
                                                      preserve_unused_token, with_offsets);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<BertTokenizerOperation> BertTokenizer(const std::shared_ptr<Vocab> &vocab,
                                                      const std::string &suffix_indicator, int32_t max_bytes_per_token,
                                                      const std::string &unknown_token, bool lower_case,
                                                      bool keep_whitespace, const NormalizeForm normalize_form,
                                                      bool preserve_unused_token, bool with_offsets) {
  auto op =
    std::make_shared<BertTokenizerOperation>(vocab, suffix_indicator, max_bytes_per_token, unknown_token, lower_case,
                                             keep_whitespace, normalize_form, preserve_unused_token, with_offsets);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<CaseFoldOperation> CaseFold() {
  auto op = std::make_shared<CaseFoldOperation>();

  return op->ValidateParams() ? op : nullptr;
}
#endif

std::shared_ptr<JiebaTokenizerOperation> JiebaTokenizer(const std::string &hmm_path, const std::string &mp_path,
                                                        const JiebaMode &mode, bool with_offsets) {
  auto op = std::make_shared<JiebaTokenizerOperation>(hmm_path, mp_path, mode, with_offsets);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<LookupOperation> Lookup(const std::shared_ptr<Vocab> &vocab,
                                        const std::optional<std::string> &unknown_token, const std::string &data_type) {
  auto op = std::make_shared<LookupOperation>(vocab, unknown_token, data_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<NgramOperation> Ngram(const std::vector<int32_t> &ngrams,
                                      const std::pair<std::string, int32_t> &left_pad,
                                      const std::pair<std::string, int32_t> &right_pad, const std::string &separator) {
  auto op = std::make_shared<NgramOperation>(ngrams, left_pad, right_pad, separator);

  return op->ValidateParams() ? op : nullptr;
}

#ifndef _WIN32
std::shared_ptr<NormalizeUTF8Operation> NormalizeUTF8(NormalizeForm normalize_form) {
  auto op = std::make_shared<NormalizeUTF8Operation>(normalize_form);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<RegexReplaceOperation> RegexReplace(std::string pattern, std::string replace, bool replace_all) {
  auto op = std::make_shared<RegexReplaceOperation>(pattern, replace, replace_all);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<RegexTokenizerOperation> RegexTokenizer(std::string delim_pattern, std::string keep_delim_pattern,
                                                        bool with_offsets) {
  auto op = std::make_shared<RegexTokenizerOperation>(delim_pattern, keep_delim_pattern, with_offsets);

  return op->ValidateParams() ? op : nullptr;
}
#endif

std::shared_ptr<SentencePieceTokenizerOperation> SentencePieceTokenizer(
  const std::shared_ptr<SentencePieceVocab> &vocab, SPieceTokenizerOutType out_type) {
  auto op = std::make_shared<SentencePieceTokenizerOperation>(vocab, out_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<SentencePieceTokenizerOperation> SentencePieceTokenizer(const std::string &vocab_path,
                                                                        SPieceTokenizerOutType out_type) {
  auto op = std::make_shared<SentencePieceTokenizerOperation>(vocab_path, out_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<SlidingWindowOperation> SlidingWindow(const int32_t width, const int32_t axis) {
  auto op = std::make_shared<SlidingWindowOperation>(width, axis);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<ToNumberOperation> ToNumber(const std::string &data_type) {
  auto op = std::make_shared<ToNumberOperation>(data_type);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<TruncateSequencePairOperation> TruncateSequencePair(int32_t max_length) {
  auto op = std::make_shared<TruncateSequencePairOperation>(max_length);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<UnicodeCharTokenizerOperation> UnicodeCharTokenizer(bool with_offsets) {
  auto op = std::make_shared<UnicodeCharTokenizerOperation>(with_offsets);

  return op->ValidateParams() ? op : nullptr;
}

#ifndef _WIN32
std::shared_ptr<UnicodeScriptTokenizerOperation> UnicodeScriptTokenizer(bool keep_whitespace, bool with_offsets) {
  auto op = std::make_shared<UnicodeScriptTokenizerOperation>(keep_whitespace, with_offsets);

  return op->ValidateParams() ? op : nullptr;
}

std::shared_ptr<WhitespaceTokenizerOperation> WhitespaceTokenizer(bool with_offsets) {
  auto op = std::make_shared<WhitespaceTokenizerOperation>(with_offsets);

  return op->ValidateParams() ? op : nullptr;
}
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
