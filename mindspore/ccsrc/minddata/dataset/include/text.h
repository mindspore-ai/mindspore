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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TEXT_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "minddata/dataset/include/constants.h"
#include "minddata/dataset/include/transforms.h"

namespace mindspore {
namespace dataset {

class Vocab;
class SentencePieceVocab;
class TensorOperation;

// Transform operations for text
namespace text {

#ifndef _WIN32
/// \brief Tokenize a scalar tensor of UTF-8 string by specific rules.
/// \notes BasicTokenizer is not supported on Windows platform yet.
class BasicTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] lower_case If true, apply CaseFold, NormalizeUTF8(NFD mode), RegexReplace operation on input text to
  ///   fold the text to lower case and strip accents characters. If false, only apply
  ///   NormalizeUTF8('normalization_form' mode) operation on input text (default=false).
  /// \param[in] keep_whitespace If true, the whitespace will be kept in out tokens (default=false).
  /// \param[in] normalize_form Used to specify a specific normalize mode. This is only effective when 'lower_case' is
  ///   false. See NormalizeUTF8 for details (default=NormalizeForm::kNone).
  /// \param[in] preserve_unused_token If true, do not split special tokens like '[CLS]', '[SEP]', '[UNK]', '[PAD]',
  ///   '[MASK]' (default=true).
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit BasicTokenizer(bool lower_case = false, bool keep_whitespace = false,
                          const NormalizeForm normalize_form = NormalizeForm::kNone, bool preserve_unused_token = true,
                          bool with_offsets = false);

  /// \brief Destructor
  ~BasicTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenizer used for Bert text process.
/// \notes BertTokenizer is not supported on Windows platform yet.
class BertTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab A Vocab object.
  /// \param[in] suffix_indicator Used to show that the subword is the last part of a word (default='##').
  /// \param[in] max_bytes_per_token Tokens exceeding this length will not be further split (default=100).
  /// \param[in] unknown_token When a token cannot be found, return the token directly if 'unknown_token' is an empty
  ///   string, else return the string specified(default='[UNK]').
  /// \param[in] lower_case If true, apply CaseFold, NormalizeUTF8(NFD mode), RegexReplace operation on input text to
  ///   fold the text to lower case and strip accents characters. If false, only apply
  ///   NormalizeUTF8('normalization_form' mode) operation on input text (default=false).
  /// \param[in] keep_whitespace If true, the whitespace will be kept in out tokens (default=false).
  /// \param[in] normalize_form Used to specify a specific normalize mode. This is only effective when 'lower_case' is
  ///   false. See NormalizeUTF8 for details (default=NormalizeForm::kNone).
  /// \param[in] preserve_unused_token If true, do not split special tokens like '[CLS]', '[SEP]', '[UNK]', '[PAD]',
  ///   '[MASK]' (default=true).
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = "##",
                         int32_t max_bytes_per_token = 100, const std::string &unknown_token = "[UNK]",
                         bool lower_case = false, bool keep_whitespace = false,
                         const NormalizeForm normalize_form = NormalizeForm::kNone, bool preserve_unused_token = true,
                         bool with_offsets = false)
      : BertTokenizer(vocab, StringToChar(suffix_indicator), max_bytes_per_token, StringToChar(unknown_token),
                      lower_case, keep_whitespace, normalize_form, preserve_unused_token, with_offsets) {}

  explicit BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                         int32_t max_bytes_per_token, const std::vector<char> &unknown_token, bool lower_case,
                         bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                         bool with_offsets);

  /// \brief Destructor
  ~BertTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply case fold operation on UTF-8 string tensor.
/// \return Shared pointer to the current TensorOperation.
class CaseFold final : public TensorTransform {
 public:
  /// \brief Constructor.
  CaseFold();

  /// \brief Destructor
  ~CaseFold() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  //// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
#endif

/// \brief Tokenize Chinese string into words based on dictionary.
/// \notes The integrity of the HMMSEgment algorithm and MPSegment algorithm files must be confirmed.
class JiebaTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hmm_path Dictionary file is used by HMMSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba.
  /// \param[in] mp_path Dictionary file is used by MPSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba.
  /// \param[in] mode Valid values can be any of [JiebaMode.MP, JiebaMode.HMM, JiebaMode.MIX](default=JiebaMode.MIX).
  ///   - JiebaMode.kMP, tokenize with MPSegment algorithm.
  ///   - JiebaMode.kHMM, tokenize with Hiddel Markov Model Segment algorithm.
  ///   - JiebaMode.kMIX, tokenize with a mix of MPSegment and HMMSegment algorithm.
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit JiebaTokenizer(const std::string &hmm_path, const std::string &mp_path,
                          const JiebaMode &mode = JiebaMode::kMix, bool with_offsets = false)
      : JiebaTokenizer(StringToChar(hmm_path), StringToChar(mp_path), mode, with_offsets) {}

  explicit JiebaTokenizer(const std::vector<char> &hmm_path, const std::vector<char> &mp_path, const JiebaMode &mode,
                          bool with_offsets);

  /// \brief Destructor
  ~JiebaTokenizer() = default;

  Status AddWord(const std::string &word, int64_t freq = 0);

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Look up a word into an id according to the input vocabulary table.
class Lookup final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a Vocab object.
  /// \param[in] unknown_token word to use for lookup if the word being looked up is out of Vocabulary (oov).
  ///   If unknown_token is oov, runtime error will be thrown. If unknown_token is {}, which means that not to
  ///    specify unknown_token when word being out of Vocabulary (default={}).
  /// \param[in] data_type type of the tensor after lookup, typically int32.
  explicit Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::string> &unknown_token = {},
                  const std::string &data_type = "int32")
      : Lookup(vocab, OptionalStringToChar(unknown_token), StringToChar(data_type)) {}

  explicit Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::vector<char>> &unknown_token,
                  const std::vector<char> &data_type);

  /// \brief Destructor
  ~Lookup() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TensorOp to generate n-gram from a 1-D string Tensor.
class Ngram final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] ngrams ngrams is a vector of positive integers. For example, if ngrams={4, 3}, then the result
  ///   would be a 4-gram followed by a 3-gram in the same tensor. If the number of words is not enough to make up
  ///   for a n-gram, an empty string will be returned.
  /// \param[in] left_pad {"pad_token", pad_width}. Padding performed on left side of the sequence. pad_width will
  ///   be capped at n-1. left_pad=("_",2) would pad left side of the sequence with "__" (default={"", 0}}).
  /// \param[in] right_pad {"pad_token", pad_width}. Padding performed on right side of the sequence.pad_width will
  ///   be capped at n-1. right_pad=("-":2) would pad right side of the sequence with "--" (default={"", 0}}).
  /// \param[in] separator Symbol used to join strings together (default=" ").
  explicit Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad = {"", 0},
                 const std::pair<std::string, int32_t> &right_pad = {"", 0}, const std::string &separator = " ")
      : Ngram(ngrams, PairStringToChar(left_pad), PairStringToChar(right_pad), StringToChar(separator)) {}

  explicit Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::vector<char>, int32_t> &left_pad,
                 const std::pair<std::vector<char>, int32_t> &right_pad, const std::vector<char> &separator);

  /// \brief Destructor
  ~Ngram() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

#ifndef _WIN32
/// \brief Apply normalize operation on UTF-8 string tensor.
class NormalizeUTF8 final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] normalize_form Valid values can be any of [NormalizeForm::kNone,NormalizeForm::kNfc,
  ///   NormalizeForm::kNfkc,
  ///   NormalizeForm::kNfd, NormalizeForm::kNfkd](default=NormalizeForm::kNfkc).
  ///   See http://unicode.org/reports/tr15/ for details.
  ///   - NormalizeForm.NONE, do nothing for input string tensor.
  ///   - NormalizeForm.NFC, normalize with Normalization Form C.
  ///   - NormalizeForm.NFKC, normalize with Normalization Form KC.
  ///   - NormalizeForm.NFD, normalize with Normalization Form D.
  ///   - NormalizeForm.NFKD, normalize with Normalization Form KD.
  explicit NormalizeUTF8(NormalizeForm normalize_form = NormalizeForm::kNfkc);

  /// \brief Destructor
  ~NormalizeUTF8() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Replace UTF-8 string tensor with 'replace' according to regular expression 'pattern'.
class RegexReplace final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] pattern The regex expression patterns.
  /// \param[in] replace The string to replace matched element.
  /// \param[in] replace_all Confirm whether to replace all. If false, only replace first matched element;
  ///   if true, replace all matched elements (default=true).
  explicit RegexReplace(std::string pattern, std::string replace, bool replace_all = true)
      : RegexReplace(StringToChar(pattern), StringToChar(replace), replace_all) {}

  explicit RegexReplace(const std::vector<char> &pattern, const std::vector<char> &replace, bool replace_all);

  /// \brief Destructor
  ~RegexReplace() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string by regex expression pattern.
class RegexTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] delim_pattern The pattern of regex delimiters.
  /// \param[in] keep_delim_pattern The string matched by 'delim_pattern' can be kept as a token if it can be
  ///   matched by 'keep_delim_pattern'. The default value is an empty string ("")
  ///   which means that delimiters will not be kept as an output token (default="").
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit RegexTokenizer(std::string delim_pattern, std::string keep_delim_pattern = "", bool with_offsets = false)
      : RegexTokenizer(StringToChar(delim_pattern), StringToChar(keep_delim_pattern), with_offsets) {}

  explicit RegexTokenizer(const std::vector<char> &delim_pattern, const std::vector<char> &keep_delim_pattern,
                          bool with_offsets);

  /// \brief Destructor
  ~RegexTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
#endif

/// \brief Tokenize scalar token or 1-D tokens to tokens by sentencepiece.
class SentencePieceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a SentencePieceVocab object.
  /// \param[in] out_type The type of output.
  SentencePieceTokenizer(const std::shared_ptr<SentencePieceVocab> &vocab,
                         mindspore::dataset::SPieceTokenizerOutType out_typee);

  /// \brief Constructor.
  /// \param[in] vocab_path vocab model file path.
  /// \param[in] out_type The type of output.
  SentencePieceTokenizer(const std::string &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type)
      : SentencePieceTokenizer(StringToChar(vocab_path), out_type) {}

  SentencePieceTokenizer(const std::vector<char> &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type);

  /// \brief Destructor
  ~SentencePieceTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief TensorOp to construct a tensor from data (only 1-D for now), where each element in the dimension
///   axis is a slice of data starting at the corresponding position, with a specified width.
class SlidingWindow final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] width The width of the window. It must be an integer and greater than zero.
  /// \param[in] axis The axis along which the sliding window is computed (default=0), axis support 0 or -1 only
  ///   for now.
  explicit SlidingWindow(const int32_t width, const int32_t axis = 0);

  /// \brief Destructor
  ~SlidingWindow() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tensor operation to convert every element of a string tensor to a number.
///   Strings are casted according to the rules specified in the following links:
///   https://en.cppreference.com/w/cpp/string/basic_string/stof,
///   https://en.cppreference.com/w/cpp/string/basic_string/stoul,
///   except that any strings which represent negative numbers cannot be cast to an unsigned integer type.
class ToNumber final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type of the tensor to be casted to. Must be a numeric type.
  explicit ToNumber(const std::string &data_type) : ToNumber(StringToChar(data_type)) {}

  explicit ToNumber(const std::vector<char> &data_type);

  /// \brief Destructor
  ~ToNumber() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Truncate a pair of rank-1 tensors such that the total length is less than max_length.
class TruncateSequencePair final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] max_length Maximum length required.
  explicit TruncateSequencePair(int32_t max_length);

  /// \brief Destructor
  ~TruncateSequencePair() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string to Unicode characters.
class UnicodeCharTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit UnicodeCharTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeCharTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

#ifndef _WIN32
/// \brief Tokenize a scalar tensor of UTF-8 string on Unicode script boundaries.
class UnicodeScriptTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] keep_whitespace If or not emit whitespace tokens (default=false).
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit UnicodeScriptTokenizer(bool keep_whitespace = false, bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeScriptTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string on ICU4C defined whitespaces.
class WhitespaceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets If or not output offsets of tokens (default=false).
  explicit WhitespaceTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~WhitespaceTokenizer() = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_TEXT_H_
