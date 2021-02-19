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

#include "minddata/dataset/text/ir/kernels/text_ir.h"

namespace mindspore {
namespace dataset {

// Transform operations for text.
namespace text {

// FUNCTIONS TO CREATE TEXT OPERATIONS
// (In alphabetical order)

#ifndef _WIN32
// BasicTokenizer
BasicTokenizer::BasicTokenizer(bool lower_case, bool keep_whitespace, const NormalizeForm normalize_form,
                               bool preserve_unused_token, bool with_offsets)
    : lower_case_(lower_case),
      keep_whitespace_(keep_whitespace),
      normalize_form_(normalize_form),
      preserve_unused_token_(preserve_unused_token),
      with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> BasicTokenizer::Parse() {
  return std::make_shared<BasicTokenizerOperation>(lower_case_, keep_whitespace_, normalize_form_,
                                                   preserve_unused_token_, with_offsets_);
}

// BertTokenizer
BertTokenizer::BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator,
                             int32_t max_bytes_per_token, const std::string &unknown_token, bool lower_case,
                             bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                             bool with_offsets)
    : vocab_(vocab),
      suffix_indicator_(suffix_indicator),
      max_bytes_per_token_(max_bytes_per_token),
      unknown_token_(unknown_token),
      lower_case_(lower_case),
      keep_whitespace_(keep_whitespace),
      normalize_form_(normalize_form),
      preserve_unused_token_(preserve_unused_token),
      with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> BertTokenizer::Parse() {
  return std::make_shared<BertTokenizerOperation>(vocab_, suffix_indicator_, max_bytes_per_token_, unknown_token_,
                                                  lower_case_, keep_whitespace_, normalize_form_,
                                                  preserve_unused_token_, with_offsets_);
}

// CaseFold
CaseFold::CaseFold() {}

std::shared_ptr<TensorOperation> CaseFold::Parse() { return std::make_shared<CaseFoldOperation>(); }
#endif

// JiebaTokenizer
JiebaTokenizer::JiebaTokenizer(const std::string &hmm_path, const std::string &mp_path, const JiebaMode &mode,
                               bool with_offsets)
    : hmm_path_(hmm_path), mp_path_(mp_path), mode_(mode), with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> JiebaTokenizer::Parse() {
  std::shared_ptr<JiebaTokenizerOperation> jieba_tokenizer =
    std::make_shared<JiebaTokenizerOperation>(hmm_path_, mp_path_, mode_, with_offsets_);
  for (auto &word : words_list_) {
    Status rc = jieba_tokenizer->AddWord(word.first, word.second);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      return {};
    }
  }
  return jieba_tokenizer;
}

Status JiebaTokenizer::AddWord(const std::string &word, int64_t freq) {
  if (word.empty()) {
    std::string err_msg = "JiebaTokenizer : The parameter word is empty or not provided.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (freq < 0) {
    std::string err_msg = "JiebaTokenizer : The parameter freq must be greater than or equal to 0.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  words_list_.emplace_back(word, freq);
  return Status::OK();
}

// Lookup
Lookup::Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::string> &unknown_token,
               const std::string &data_type)
    : vocab_(vocab), unknown_token_(unknown_token), data_type_(data_type) {}

std::shared_ptr<TensorOperation> Lookup::Parse() {
  return std::make_shared<LookupOperation>(vocab_, unknown_token_, data_type_);
}

// Ngram
Ngram::Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad,
             const std::pair<std::string, int32_t> &right_pad, const std::string &separator)
    : ngrams_(ngrams), left_pad_(left_pad), right_pad_(right_pad), separator_(separator) {}

std::shared_ptr<TensorOperation> Ngram::Parse() {
  return std::make_shared<NgramOperation>(ngrams_, left_pad_, right_pad_, separator_);
}

#ifndef _WIN32
// NormalizeUTF8
NormalizeUTF8::NormalizeUTF8(NormalizeForm normalize_form) : normalize_form_(normalize_form) {}

std::shared_ptr<TensorOperation> NormalizeUTF8::Parse() {
  return std::make_shared<NormalizeUTF8Operation>(normalize_form_);
}

// RegexReplace
RegexReplace::RegexReplace(std::string pattern, std::string replace, bool replace_all)
    : pattern_(pattern), replace_(replace), replace_all_(replace_all) {}

std::shared_ptr<TensorOperation> RegexReplace::Parse() {
  return std::make_shared<RegexReplaceOperation>(pattern_, replace_, replace_all_);
}

// RegexTokenizer
RegexTokenizer::RegexTokenizer(std::string delim_pattern, std::string keep_delim_pattern, bool with_offsets)
    : delim_pattern_(delim_pattern), keep_delim_pattern_(keep_delim_pattern), with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> RegexTokenizer::Parse() {
  return std::make_shared<RegexTokenizerOperation>(delim_pattern_, keep_delim_pattern_, with_offsets_);
}
#endif

// SentencePieceTokenizer
SentencePieceTokenizer::SentencePieceTokenizer(const std::shared_ptr<SentencePieceVocab> &vocab,
                                               SPieceTokenizerOutType out_type)
    : vocab_(vocab), out_type_(out_type) {}

SentencePieceTokenizer::SentencePieceTokenizer(const std::string &vocab_path, SPieceTokenizerOutType out_type)
    : vocab_path_(vocab_path), out_type_(out_type) {}

std::shared_ptr<TensorOperation> SentencePieceTokenizer::Parse() {
  if (vocab_ != nullptr) {
    return std::make_shared<SentencePieceTokenizerOperation>(vocab_, out_type_);
  } else {
    return std::make_shared<SentencePieceTokenizerOperation>(vocab_path_, out_type_);
  }
}

// SlidingWindow
SlidingWindow::SlidingWindow(const int32_t width, const int32_t axis) : width_(width), axis_(axis) {}

std::shared_ptr<TensorOperation> SlidingWindow::Parse() {
  return std::make_shared<SlidingWindowOperation>(width_, axis_);
}

// ToNumber
ToNumber::ToNumber(const std::string &data_type) : data_type_(data_type) {}

std::shared_ptr<TensorOperation> ToNumber::Parse() { return std::make_shared<ToNumberOperation>(data_type_); }

// TruncateSequencePair
TruncateSequencePair::TruncateSequencePair(int32_t max_length) : max_length_(max_length) {}

std::shared_ptr<TensorOperation> TruncateSequencePair::Parse() {
  return std::make_shared<TruncateSequencePairOperation>(max_length_);
}

// UnicodeCharTokenizer
UnicodeCharTokenizer::UnicodeCharTokenizer(bool with_offsets) : with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> UnicodeCharTokenizer::Parse() {
  return std::make_shared<UnicodeCharTokenizerOperation>(with_offsets_);
}

#ifndef _WIN32
// UnicodeScriptTokenizer
UnicodeScriptTokenizer::UnicodeScriptTokenizer(bool keep_whitespace, bool with_offsets)
    : keep_whitespace_(keep_whitespace), with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> UnicodeScriptTokenizer::Parse() {
  return std::make_shared<UnicodeScriptTokenizerOperation>(keep_whitespace_, with_offsets_);
}

// WhitespaceTokenizer
WhitespaceTokenizer::WhitespaceTokenizer(bool with_offsets) : with_offsets_(with_offsets) {}

std::shared_ptr<TensorOperation> WhitespaceTokenizer::Parse() {
  return std::make_shared<WhitespaceTokenizerOperation>(with_offsets_);
}
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
