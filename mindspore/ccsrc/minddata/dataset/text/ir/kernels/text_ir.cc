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

#include "minddata/dataset/text/ir/kernels/text_ir.h"

#ifndef _WIN32
#include "minddata/dataset/text/kernels/basic_tokenizer_op.h"
#include "minddata/dataset/text/kernels/bert_tokenizer_op.h"
#include "minddata/dataset/text/kernels/case_fold_op.h"
#endif
#include "minddata/dataset/text/kernels/jieba_tokenizer_op.h"
#include "minddata/dataset/text/kernels/lookup_op.h"
#include "minddata/dataset/text/kernels/ngram_op.h"
#ifndef _WIN32
#include "minddata/dataset/text/kernels/normalize_utf8_op.h"
#include "minddata/dataset/text/kernels/regex_replace_op.h"
#include "minddata/dataset/text/kernels/regex_tokenizer_op.h"
#endif
#include "minddata/dataset/text/kernels/sentence_piece_tokenizer_op.h"
#include "minddata/dataset/text/kernels/sliding_window_op.h"
#include "minddata/dataset/text/kernels/to_number_op.h"
#include "minddata/dataset/text/kernels/truncate_sequence_pair_op.h"
#include "minddata/dataset/text/kernels/unicode_char_tokenizer_op.h"
#ifndef _WIN32
#include "minddata/dataset/text/kernels/unicode_script_tokenizer_op.h"
#include "minddata/dataset/text/kernels/whitespace_tokenizer_op.h"
#endif
#include "minddata/dataset/core/data_type.h"
#include "minddata/dataset/util/path.h"

#include "minddata/dataset/text/ir/validators.h"

namespace mindspore {
namespace dataset {
// Transform operations for text.
namespace text {
/* ####################################### Derived TensorOperation classes ################################# */

// (In alphabetical order)

#ifndef _WIN32
// BasicTokenizerOperation
BasicTokenizerOperation::BasicTokenizerOperation(bool lower_case, bool keep_whitespace,
                                                 const NormalizeForm normalize_form, bool preserve_unused_token,
                                                 bool with_offsets)
    : lower_case_(lower_case),
      keep_whitespace_(keep_whitespace),
      normalize_form_(normalize_form),
      preserve_unused_token_(preserve_unused_token),
      with_offsets_(with_offsets) {}

Status BasicTokenizerOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> BasicTokenizerOperation::Build() {
  std::shared_ptr<BasicTokenizerOp> tensor_op = std::make_shared<BasicTokenizerOp>(
    lower_case_, keep_whitespace_, normalize_form_, preserve_unused_token_, with_offsets_);
  return tensor_op;
}

// BertTokenizerOperation
BertTokenizerOperation::BertTokenizerOperation(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator,
                                               int32_t max_bytes_per_token, const std::string &unknown_token,
                                               bool lower_case, bool keep_whitespace,
                                               const NormalizeForm normalize_form, bool preserve_unused_token,
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

BertTokenizerOperation::~BertTokenizerOperation() = default;

Status BertTokenizerOperation::ValidateParams() {
  if (vocab_ == nullptr) {
    std::string err_msg = "BertTokenizer: vocab object type is incorrect or null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (max_bytes_per_token_ < 0) {
    std::string err_msg = "BertTokenizer : The parameter max_bytes_per_token must be greater than or equal to 0: " +
                          std::to_string(max_bytes_per_token_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> BertTokenizerOperation::Build() {
  std::shared_ptr<BertTokenizerOp> tensor_op =
    std::make_shared<BertTokenizerOp>(vocab_, suffix_indicator_, max_bytes_per_token_, unknown_token_, lower_case_,
                                      keep_whitespace_, normalize_form_, preserve_unused_token_, with_offsets_);
  return tensor_op;
}

// CaseFoldOperation
Status CaseFoldOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> CaseFoldOperation::Build() {
  std::shared_ptr<CaseFoldOp> tensor_op = std::make_shared<CaseFoldOp>();
  return tensor_op;
}
#endif

// JiebaTokenizerOperation
JiebaTokenizerOperation::JiebaTokenizerOperation(const std::string &hmm_path, const std::string &mp_path,
                                                 const JiebaMode &mode, bool with_offsets)
    : hmm_path_(hmm_path), mp_path_(mp_path), mode_(mode), with_offsets_(with_offsets) {}

Status JiebaTokenizerOperation::ValidateParams() {
  if (hmm_path_.empty()) {
    std::string err_msg = "JiebaTokenizer: The dict of HMMSegment in cppjieba is not provided.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (mp_path_.empty()) {
    std::string err_msg = "JiebaTokenizer: The dict of MPSegment in cppjieba is not provided.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  RETURN_IF_NOT_OK(ValidateTokenizerDirParam("JiebaTokenizer", hmm_path_));
  RETURN_IF_NOT_OK(ValidateTokenizerDirParam("JiebaTokenizer", mp_path_));
  return Status::OK();
}

std::shared_ptr<TensorOp> JiebaTokenizerOperation::Build() {
  std::shared_ptr<JiebaTokenizerOp> tensor_op =
    std::make_shared<JiebaTokenizerOp>(hmm_path_, mp_path_, mode_, with_offsets_);
  for (auto &word : words_list_) {
    Status rc = tensor_op->AddWord(word.first, word.second);
    if (rc.IsError()) {
      MS_LOG(ERROR) << rc;
      return {};
    }
  }
  return tensor_op;
}

Status JiebaTokenizerOperation::AddWord(const std::string &word, int64_t freq) {
  words_list_.emplace_back(word, freq);
  return Status::OK();
}

// LookupOperation
LookupOperation::LookupOperation(const std::shared_ptr<Vocab> &vocab, const std::optional<std::string> &unknown_token,
                                 const std::string &data_type)
    : vocab_(vocab), unknown_token_(unknown_token), default_id_(Vocab::kNoTokenExists), data_type_(data_type) {}

LookupOperation::~LookupOperation() = default;

Status LookupOperation::ValidateParams() {
  if (vocab_ == nullptr) {
    std::string err_msg = "Lookup: vocab object type is incorrect or null.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  if (unknown_token_ != std::nullopt) {
    default_id_ = vocab_->Lookup(*unknown_token_);
    if (default_id_ == Vocab::kNoTokenExists) {
      std::string err_msg = "Lookup: \"" + *unknown_token_ + "\" doesn't exist in vocab.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }

  if (!IsTypeNumeric(data_type_)) {
    std::string err_msg = "Lookup does not support a string to string mapping, data_type can only be numeric.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> LookupOperation::Build() {
  std::shared_ptr<LookupOp> tensor_op = std::make_shared<LookupOp>(vocab_, default_id_, DataType(data_type_));
  return tensor_op;
}

// NgramOperation
NgramOperation::NgramOperation(const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad,
                               const std::pair<std::string, int32_t> &right_pad, const std::string &separator)
    : ngrams_(ngrams), left_pad_(left_pad), right_pad_(right_pad), separator_(separator) {}

Status NgramOperation::ValidateParams() {
  if (ngrams_.size() == 0) {
    std::string err_msg = "Ngram : Container cannot be empty.";
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  } else {
    for (int32_t i = 0; i < ngrams_.size(); ++i) {
      if (ngrams_[i] <= 0) {
        std::string err_msg =
          "Ngram : The value of ngrams vector must be greater than 0: " + std::to_string(ngrams_[i]);
        MS_LOG(ERROR) << err_msg;
        RETURN_STATUS_SYNTAX_ERROR(err_msg);
      }
    }
  }

  if (left_pad_.second < 0) {
    std::string err_msg =
      "Ngram : The second parameter pad_width in left_pad vector must be greater than or equal to 0: " +
      std::to_string(left_pad_.second);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  if (right_pad_.second < 0) {
    std::string err_msg =
      "Ngram : The second parameter pad_width in right_pad vector must be greater than or equal to 0: " +
      std::to_string(right_pad_.second);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> NgramOperation::Build() {
  int32_t l_len = left_pad_.second;
  int32_t r_len = right_pad_.second;
  std::string l_pad = left_pad_.first;
  std::string r_pad = right_pad_.first;
  std::shared_ptr<NgramOp> tensor_op = std::make_shared<NgramOp>(ngrams_, l_len, r_len, l_pad, r_pad, separator_);
  return tensor_op;
}

#ifndef _WIN32
// NormalizeUTF8Operation
NormalizeUTF8Operation::NormalizeUTF8Operation(NormalizeForm normalize_form) : normalize_form_(normalize_form) {}

Status NormalizeUTF8Operation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> NormalizeUTF8Operation::Build() {
  std::shared_ptr<NormalizeUTF8Op> tensor_op = std::make_shared<NormalizeUTF8Op>(normalize_form_);
  return tensor_op;
}

// RegexReplaceOperation
RegexReplaceOperation::RegexReplaceOperation(std::string pattern, std::string replace, bool replace_all)
    : pattern_(pattern), replace_(replace), replace_all_(replace_all) {}

Status RegexReplaceOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RegexReplaceOperation::Build() {
  std::shared_ptr<RegexReplaceOp> tensor_op = std::make_shared<RegexReplaceOp>(pattern_, replace_, replace_all_);
  return tensor_op;
}

// RegexTokenizerOperation
RegexTokenizerOperation::RegexTokenizerOperation(std::string delim_pattern, std::string keep_delim_pattern,
                                                 bool with_offsets)
    : delim_pattern_(delim_pattern), keep_delim_pattern_(keep_delim_pattern), with_offsets_(with_offsets) {}

Status RegexTokenizerOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> RegexTokenizerOperation::Build() {
  std::shared_ptr<RegexTokenizerOp> tensor_op =
    std::make_shared<RegexTokenizerOp>(delim_pattern_, keep_delim_pattern_, with_offsets_);
  return tensor_op;
}
#endif

// SentencePieceTokenizerOperation
SentencePieceTokenizerOperation::~SentencePieceTokenizerOperation() = default;

SentencePieceTokenizerOperation::SentencePieceTokenizerOperation(const std::shared_ptr<SentencePieceVocab> &vocab,
                                                                 SPieceTokenizerOutType out_type)
    : vocab_(vocab), vocab_path_(std::string()), load_type_(SPieceTokenizerLoadType::kModel), out_type_(out_type) {}

SentencePieceTokenizerOperation::SentencePieceTokenizerOperation(const std::string &vocab_path,
                                                                 SPieceTokenizerOutType out_type)
    : vocab_(nullptr), vocab_path_(vocab_path), load_type_(SPieceTokenizerLoadType::kFile), out_type_(out_type) {}

Status SentencePieceTokenizerOperation::ValidateParams() {
  if (load_type_ == SPieceTokenizerLoadType::kModel) {
    if (vocab_ == nullptr) {
      std::string err_msg = "SentencePieceTokenizer: vocab object type is incorrect or null.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  } else {
    Path vocab_file(vocab_path_);
    if (!vocab_file.Exists() || vocab_file.IsDirectory()) {
      std::string err_msg = "SentencePieceTokenizer : vocab file: [" + vocab_path_ + "] is invalid or does not exist.";
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
    if (access(vocab_file.toString().c_str(), R_OK) == -1) {
      std::string err_msg = "SentencePieceTokenizer : no access to specified dataset file: " + vocab_path_;
      MS_LOG(ERROR) << err_msg;
      RETURN_STATUS_SYNTAX_ERROR(err_msg);
    }
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SentencePieceTokenizerOperation::Build() {
  std::shared_ptr<SentencePieceTokenizerOp> tensor_op;
  if (load_type_ == SPieceTokenizerLoadType::kModel) {
    tensor_op = std::make_shared<SentencePieceTokenizerOp>(vocab_, load_type_, out_type_);
  } else {
    Path vocab_file(vocab_path_);
    std::string model_path = vocab_file.ParentPath();
    std::string model_filename = vocab_file.Basename();
    tensor_op = std::make_shared<SentencePieceTokenizerOp>(model_path, model_filename, load_type_, out_type_);
  }
  return tensor_op;
}

// SlidingWindowOperation
SlidingWindowOperation::SlidingWindowOperation(const int32_t width, const int32_t axis) : width_(width), axis_(axis) {}

Status SlidingWindowOperation::ValidateParams() {
  if (width_ < 1) {
    std::string err_msg =
      "SlidingWindow : The parameter width must be greater than or equal to 1: " + std::to_string(width_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }
  return Status::OK();
}

std::shared_ptr<TensorOp> SlidingWindowOperation::Build() {
  std::shared_ptr<SlidingWindowOp> tensor_op = std::make_shared<SlidingWindowOp>(static_cast<uint32_t>(width_), axis_);
  return tensor_op;
}

// ToNumberOperation
ToNumberOperation::ToNumberOperation(std::string data_type) : data_type_(data_type) {}

Status ToNumberOperation::ValidateParams() {
  if (!IsTypeNumeric(data_type_) || IsTypeBoolean(data_type_)) {
    std::string err_msg = "ToNumber : The parameter data_type must be a numeric type, got: " + data_type_;
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> ToNumberOperation::Build() {
  std::shared_ptr<ToNumberOp> tensor_op = std::make_shared<ToNumberOp>(data_type_);
  return tensor_op;
}

// TruncateSequencePairOperation
TruncateSequencePairOperation::TruncateSequencePairOperation(int32_t max_length) : max_length_(max_length) {}

Status TruncateSequencePairOperation::ValidateParams() {
  if (max_length_ < 0) {
    std::string err_msg = "TruncateSequencePair : The parameter max_length must be greater than or equal to 0: " +
                          std::to_string(max_length_);
    MS_LOG(ERROR) << err_msg;
    RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  return Status::OK();
}

std::shared_ptr<TensorOp> TruncateSequencePairOperation::Build() {
  std::shared_ptr<TruncateSequencePairOp> tensor_op = std::make_shared<TruncateSequencePairOp>(max_length_);
  return tensor_op;
}

// UnicodeCharTokenizerOperation
UnicodeCharTokenizerOperation::UnicodeCharTokenizerOperation(bool with_offsets) : with_offsets_(with_offsets) {}

Status UnicodeCharTokenizerOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> UnicodeCharTokenizerOperation::Build() {
  std::shared_ptr<UnicodeCharTokenizerOp> tensor_op = std::make_shared<UnicodeCharTokenizerOp>(with_offsets_);
  return tensor_op;
}

#ifndef _WIN32
// UnicodeScriptTokenizerOperation
UnicodeScriptTokenizerOperation::UnicodeScriptTokenizerOperation(bool keep_whitespace, bool with_offsets)
    : keep_whitespace_(keep_whitespace), with_offsets_(with_offsets) {}

Status UnicodeScriptTokenizerOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> UnicodeScriptTokenizerOperation::Build() {
  std::shared_ptr<UnicodeScriptTokenizerOp> tensor_op =
    std::make_shared<UnicodeScriptTokenizerOp>(keep_whitespace_, with_offsets_);
  return tensor_op;
}

// WhitespaceTokenizerOperation
WhitespaceTokenizerOperation::WhitespaceTokenizerOperation(bool with_offsets) : with_offsets_(with_offsets) {}

Status WhitespaceTokenizerOperation::ValidateParams() { return Status::OK(); }

std::shared_ptr<TensorOp> WhitespaceTokenizerOperation::Build() {
  std::shared_ptr<WhitespaceTokenizerOp> tensor_op = std::make_shared<WhitespaceTokenizerOp>(with_offsets_);
  return tensor_op;
}
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
