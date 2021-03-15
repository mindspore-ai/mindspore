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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_KERNELS_TEXT_IR_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_KERNELS_TEXT_IR_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/kernels/ir/tensor_operation.h"

namespace mindspore {
namespace dataset {
class Vocab;
class SentencePieceVocab;

// Transform operations for text
namespace text {
// Char arrays storing name of corresponding classes (in alphabetical order)
constexpr char kBasicTokenizerOperation[] = "BasicTokenizer";
constexpr char kBertTokenizerOperation[] = "BertTokenizer";
constexpr char kCaseFoldOperation[] = "CaseFold";
constexpr char kJiebaTokenizerOperation[] = "JiebaTokenizer";
constexpr char kLookupOperation[] = "Lookup";
constexpr char kNgramOperation[] = "Ngram";
constexpr char kNormalizeUTF8Operation[] = "NormalizeUTF8";
constexpr char kRegexReplaceOperation[] = "RegexReplace";
constexpr char kRegexTokenizerOperation[] = "RegexTokenizer";
constexpr char kSentencepieceTokenizerOperation[] = "SentencepieceTokenizer";
constexpr char kSlidingWindowOperation[] = "SlidingWindow";
constexpr char kToNumberOperation[] = "ToNumber";
constexpr char kTruncateSequencePairOperation[] = "TruncateSequencePair";
constexpr char kUnicodeCharTokenizerOperation[] = "UnicodeCharTokenizer";
constexpr char kUnicodeScriptTokenizerOperation[] = "UnicodeScriptTokenizer";
constexpr char kWhitespaceTokenizerOperation[] = "WhitespaceTokenizer";

/* ####################################### Derived TensorOperation classes ################################# */

#ifndef _WIN32
class BasicTokenizerOperation : public TensorOperation {
 public:
  BasicTokenizerOperation(bool lower_case, bool keep_whitespace, const NormalizeForm normalize_form,
                          bool preserve_unused_token, bool with_offsets);

  ~BasicTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBasicTokenizerOperation; }

 private:
  bool lower_case_;
  bool keep_whitespace_;
  NormalizeForm normalize_form_;
  bool preserve_unused_token_;
  bool with_offsets_;
};

class BertTokenizerOperation : public TensorOperation {
 public:
  BertTokenizerOperation(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator,
                         int32_t max_bytes_per_token, const std::string &unknown_token, bool lower_case,
                         bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                         bool with_offsets);

  ~BertTokenizerOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kBertTokenizerOperation; }

 private:
  std::shared_ptr<Vocab> vocab_;
  std::string suffix_indicator_;
  int32_t max_bytes_per_token_;
  std::string unknown_token_;
  bool lower_case_;
  bool keep_whitespace_;
  NormalizeForm normalize_form_;
  bool preserve_unused_token_;
  bool with_offsets_;
};

class CaseFoldOperation : public TensorOperation {
 public:
  CaseFoldOperation() = default;

  ~CaseFoldOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kCaseFoldOperation; }
};
#endif

class JiebaTokenizerOperation : public TensorOperation {
 public:
  explicit JiebaTokenizerOperation(const std::string &hmm_path, const std::string &mp_path, const JiebaMode &mode,
                                   bool with_offsets);

  ~JiebaTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kJiebaTokenizerOperation; }

  Status AddWord(const std::string &word, int64_t freq = 0);

 private:
  std::string hmm_path_;
  std::string mp_path_;
  JiebaMode mode_;
  bool with_offsets_;
  std::vector<std::pair<std::string, int64_t>> words_list_;
};

class LookupOperation : public TensorOperation {
 public:
  explicit LookupOperation(const std::shared_ptr<Vocab> &vocab, const std::optional<std::string> &unknown_token,
                           const std::string &data_type);

  ~LookupOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kLookupOperation; }

 private:
  std::shared_ptr<Vocab> vocab_;
  std::optional<std::string> unknown_token_;
  int32_t default_id_;
  std::string data_type_;
};

class NgramOperation : public TensorOperation {
 public:
  explicit NgramOperation(const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad,
                          const std::pair<std::string, int32_t> &right_pad, const std::string &separator);

  ~NgramOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kNgramOperation; }

 private:
  std::vector<int32_t> ngrams_;
  std::pair<std::string, int32_t> left_pad_;
  std::pair<std::string, int32_t> right_pad_;
  std::string separator_;
};

#ifndef _WIN32
class NormalizeUTF8Operation : public TensorOperation {
 public:
  explicit NormalizeUTF8Operation(NormalizeForm normalize_form);

  ~NormalizeUTF8Operation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kNormalizeUTF8Operation; }

 private:
  NormalizeForm normalize_form_;
};

class RegexReplaceOperation : public TensorOperation {
 public:
  RegexReplaceOperation(std::string pattern, std::string replace, bool replace_all);

  ~RegexReplaceOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRegexReplaceOperation; }

 private:
  std::string pattern_;
  std::string replace_;
  bool replace_all_;
};

class RegexTokenizerOperation : public TensorOperation {
 public:
  explicit RegexTokenizerOperation(std::string delim_pattern, std::string keep_delim_pattern, bool with_offsets);

  ~RegexTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kRegexTokenizerOperation; }

 private:
  std::string delim_pattern_;
  std::string keep_delim_pattern_;
  bool with_offsets_;
};
#endif

class SentencePieceTokenizerOperation : public TensorOperation {
 public:
  SentencePieceTokenizerOperation(const std::shared_ptr<SentencePieceVocab> &vocab, SPieceTokenizerOutType out_type);

  SentencePieceTokenizerOperation(const std::string &vocab_path, SPieceTokenizerOutType out_type);

  ~SentencePieceTokenizerOperation();

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSentencepieceTokenizerOperation; }

 private:
  std::shared_ptr<SentencePieceVocab> vocab_;
  std::string vocab_path_;
  SPieceTokenizerLoadType load_type_;
  SPieceTokenizerOutType out_type_;
};

class SlidingWindowOperation : public TensorOperation {
 public:
  explicit SlidingWindowOperation(const int32_t width, const int32_t axis);

  ~SlidingWindowOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kSlidingWindowOperation; }

 private:
  int32_t width_;
  int32_t axis_;
};

class ToNumberOperation : public TensorOperation {
 public:
  explicit ToNumberOperation(std::string data_type);

  ~ToNumberOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kToNumberOperation; }

 private:
  std::string data_type_;
};

class TruncateSequencePairOperation : public TensorOperation {
 public:
  explicit TruncateSequencePairOperation(int32_t max_length);

  ~TruncateSequencePairOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kTruncateSequencePairOperation; }

 private:
  int32_t max_length_;
};

class UnicodeCharTokenizerOperation : public TensorOperation {
 public:
  explicit UnicodeCharTokenizerOperation(bool with_offsets);

  ~UnicodeCharTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUnicodeCharTokenizerOperation; }

 private:
  bool with_offsets_;
};

#ifndef _WIN32
class UnicodeScriptTokenizerOperation : public TensorOperation {
 public:
  explicit UnicodeScriptTokenizerOperation(bool keep_whitespace, bool with_offsets);

  ~UnicodeScriptTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kUnicodeScriptTokenizerOperation; }

 private:
  bool keep_whitespace_;
  bool with_offsets_;
};

class WhitespaceTokenizerOperation : public TensorOperation {
 public:
  explicit WhitespaceTokenizerOperation(bool with_offsets);

  ~WhitespaceTokenizerOperation() = default;

  std::shared_ptr<TensorOp> Build() override;

  Status ValidateParams() override;

  std::string Name() const override { return kWhitespaceTokenizerOperation; }

 private:
  bool with_offsets_;
};
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_IR_KERNELS_TEXT_IR_H_
