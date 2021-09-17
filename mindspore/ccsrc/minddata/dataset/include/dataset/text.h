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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TEXT_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TEXT_H_

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {

class Vocab;
class SentencePieceVocab;
class TensorOperation;

// Transform operations for text
namespace text {

#ifndef _WIN32
/// \brief Tokenize a scalar tensor of UTF-8 string by specific rules.
/// \note BasicTokenizer is not supported on the Windows platform yet.
class BasicTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] lower_case If true, apply CaseFold, NormalizeUTF8 (NFD mode) and RegexReplace operations to
  ///    the input text to fold the text to lower case and strip accents characters. If false, only apply
  ///    the NormalizeUTF8('normalization_form' mode) operation to the input text (default=false).
  /// \param[in] keep_whitespace If true, the whitespace will be kept in output tokens (default=false).
  /// \param[in] normalize_form This parameter is used to specify a specific normalize mode. This is only effective
  ///    when 'lower_case' is false. See NormalizeUTF8 for details (default=NormalizeForm::kNone).
  /// \param[in] preserve_unused_token If true, do not split special tokens like '[CLS]', '[SEP]', '[UNK]', '[PAD]' and
  ///    '[MASK]' (default=true).
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  explicit BasicTokenizer(bool lower_case = false, bool keep_whitespace = false,
                          const NormalizeForm normalize_form = NormalizeForm::kNone, bool preserve_unused_token = true,
                          bool with_offsets = false);

  /// \brief Destructor
  ~BasicTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief A tokenizer used for Bert text process.
/// \note BertTokenizer is not supported on the Windows platform yet.
class BertTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab A Vocab object.
  /// \param[in] suffix_indicator This parameter is used to show that the sub-word
  ///    is the last part of a word (default='##').
  /// \param[in] max_bytes_per_token Tokens exceeding this length will not be further split (default=100).
  /// \param[in] unknown_token When a token cannot be found, return the token directly if 'unknown_token' is an empty
  ///    string, else return the specified string (default='[UNK]').
  /// \param[in] lower_case If true, apply CaseFold, NormalizeUTF8 (NFD mode) and RegexReplace operations to
  ///    the input text to fold the text to lower case and strip accents characters. If false, only apply
  ///    the NormalizeUTF8('normalization_form' mode) operation to the input text (default=false).
  /// \param[in] keep_whitespace If true, the whitespace will be kept in output tokens (default=false).
  /// \param[in] normalize_form This parameter is used to specify a specific normalize mode. This is only effective
  ///    when 'lower_case' is false. See NormalizeUTF8 for details (default=NormalizeForm::kNone).
  /// \param[in] preserve_unused_token If true, do not split special tokens like '[CLS]', '[SEP]', '[UNK]', '[PAD]' and
  ///   '[MASK]' (default=true).
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  explicit BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = "##",
                         int32_t max_bytes_per_token = 100, const std::string &unknown_token = "[UNK]",
                         bool lower_case = false, bool keep_whitespace = false,
                         const NormalizeForm normalize_form = NormalizeForm::kNone, bool preserve_unused_token = true,
                         bool with_offsets = false)
      : BertTokenizer(vocab, StringToChar(suffix_indicator), max_bytes_per_token, StringToChar(unknown_token),
                      lower_case, keep_whitespace, normalize_form, preserve_unused_token, with_offsets) {}
  /// \brief Constructor.
  /// \param[in] vocab A Vocab object.
  /// \param[in] suffix_indicator This parameter is used to show that the sub-word
  ///    is the last part of a word (default='##').
  /// \param[in] max_bytes_per_token Tokens exceeding this length will not be further split (default=100).
  /// \param[in] unknown_token When a token cannot be found, return the token directly if 'unknown_token' is an empty
  ///    string, else return the specified string (default='[UNK]').
  /// \param[in] lower_case If true, apply CaseFold, NormalizeUTF8 (NFD mode) and RegexReplace operations to
  ///    the input text to fold the text to lower case and strip accents characters. If false, only apply
  ///    the NormalizeUTF8('normalization_form' mode) operation to the input text (default=false).
  /// \param[in] keep_whitespace If true, the whitespace will be kept in output tokens (default=false).
  /// \param[in] normalize_form This parameter is used to specify a specific normalize mode. This is only effective
  ///    when 'lower_case' is false. See NormalizeUTF8 for details (default=NormalizeForm::kNone).
  /// \param[in] preserve_unused_token If true, do not split special tokens like '[CLS]', '[SEP]', '[UNK]', '[PAD]' and
  ///   '[MASK]' (default=true).
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  BertTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                int32_t max_bytes_per_token, const std::vector<char> &unknown_token, bool lower_case,
                bool keep_whitespace, const NormalizeForm normalize_form, bool preserve_unused_token,
                bool with_offsets);

  /// \brief Destructor
  ~BertTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply case fold operation on UTF-8 string tensors.
class CaseFold final : public TensorTransform {
 public:
  /// \brief Constructor.
  CaseFold();

  /// \brief Destructor
  ~CaseFold() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
#endif

/// \brief Tokenize a Chinese string into words based on the dictionary.
/// \note The integrity of the HMMSegment algorithm and MPSegment algorithm files must be confirmed.
class JiebaTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] hmm_path Dictionary file is used by the HMMSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba (https://github.com/yanyiwu/cppjieba).
  /// \param[in] mp_path Dictionary file is used by the MPSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba (https://github.com/yanyiwu/cppjieba).
  /// \param[in] mode Valid values can be any of JiebaMode.kMP, JiebaMode.kHMM and JiebaMode.kMIX
  ///   (default=JiebaMode.kMIX).
  ///   - JiebaMode.kMP, tokenizes with MPSegment algorithm.
  ///   - JiebaMode.kHMM, tokenizes with Hidden Markov Model Segment algorithm.
  ///   - JiebaMode.kMIX, tokenizes with a mix of MPSegment and HMMSegment algorithms.
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  JiebaTokenizer(const std::string &hmm_path, const std::string &mp_path, const JiebaMode &mode = JiebaMode::kMix,
                 bool with_offsets = false)
      : JiebaTokenizer(StringToChar(hmm_path), StringToChar(mp_path), mode, with_offsets) {}

  /// \brief Constructor.
  /// \param[in] hmm_path Dictionary file is used by the HMMSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba (https://github.com/yanyiwu/cppjieba).
  /// \param[in] mp_path Dictionary file is used by the MPSegment algorithm. The dictionary can be obtained on the
  ///   official website of cppjieba (https://github.com/yanyiwu/cppjieba).
  /// \param[in] mode Valid values can be any of JiebaMode.kMP, JiebaMode.kHMM and JiebaMode.kMIX
  ///   (default=JiebaMode.kMIX).
  ///   - JiebaMode.kMP, tokenizes with MPSegment algorithm.
  ///   - JiebaMode.kHMM, tokenizes with Hidden Markov Model Segment algorithm.
  ///   - JiebaMode.kMIX, tokenizes with a mix of MPSegment and HMMSegment algorithms.
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  JiebaTokenizer(const std::vector<char> &hmm_path, const std::vector<char> &mp_path, const JiebaMode &mode,
                 bool with_offsets);

  /// \brief Destructor
  ~JiebaTokenizer() = default;

  /// \brief Add a user defined word to the JiebaTokenizer's dictionary.
  /// \param[in] word The word to be added to the JiebaTokenizer instance.
  ///   The added word will not be written into the built-in dictionary on disk.
  /// \param[in] freq The frequency of the word to be added. The higher the frequency,
  ///   the better chance the word will be tokenized (default=None, use default frequency).
  /// \return Status error code, returns OK if no error is encountered.
  Status AddWord(const std::string &word, int64_t freq = 0) { return AddWordChar(StringToChar(word), freq); }

  /// \brief Add a user defined dictionary of word-freq pairs to the JiebaTokenizer's dictionary.
  /// \param[in] user_dict Vector of word-freq pairs to be added to the JiebaTokenizer's dictionary.
  /// \return Status error code, returns OK if no error is encountered.
  Status AddDict(const std::vector<std::pair<std::string, int64_t>> &user_dict) {
    return AddDictChar(PairStringInt64ToPairCharInt64(user_dict));
  }

  /// \brief Add user defined dictionary of word-freq pairs to the JiebaTokenizer's dictionary from a file.
  ///   Only valid word-freq pairs in user defined file will be added into the dictionary.
  ///   Rows containing invalid inputs will be ignored, no error nor warning status is returned.
  /// \param[in] file_path Path to the dictionary which includes user defined word-freq pairs.
  /// \return Status error code, returns OK if no error is encountered.
  Status AddDict(const std::string &file_path) { return AddDictChar(StringToChar(file_path)); }

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  /// \brief Parser user defined words by files.
  /// \param[in] file_path Path to the user defined file.
  /// \param[in] user_dict Vector of word-freq pairs extracted from the user defined file.
  Status ParserFile(const std::string &file_path, std::vector<std::pair<std::string, int64_t>> *const user_dict);

  /// \brief Used to translate all API strings to vector of char and reverse.
  Status AddWordChar(const std::vector<char> &word, int64_t freq = 0);

  /// \brief Used to translate all API strings to vector of char and reverse.
  Status AddDictChar(const std::vector<std::pair<std::vector<char>, int64_t>> &user_dict);

  /// \brief Used to translate all API strings to vector of char and reverse.
  Status AddDictChar(const std::vector<char> &file_path);

  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Look up a word into an id according to the input vocabulary table.
class Lookup final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a Vocab object.
  /// \param[in] unknown_token Word is used for lookup. In case of the word is out of vocabulary (OOV),
  ///    the result of lookup will be replaced to unknown_token. If the unknown_token is not specified or it is OOV,
  ///    runtime error will be thrown (default={}, means no unknown_token is specified).
  /// \param[in] data_type mindspore::DataType of the tensor after lookup; must be numeric, including bool.
  ///   (default=mindspore::DataType::kNumberTypeInt32).
  explicit Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::string> &unknown_token = {},
                  mindspore::DataType data_type = mindspore::DataType::kNumberTypeInt32) {
    std::optional<std::vector<char>> unknown_token_c = std::nullopt;
    if (unknown_token != std::nullopt) {
      unknown_token_c = std::vector<char>(unknown_token->begin(), unknown_token->end());
    }
    new (this) Lookup(vocab, unknown_token_c, data_type);
  }

  /// \brief Constructor.
  /// \param[in] vocab a Vocab object.
  /// \param[in] unknown_token Word is used for lookup. In case of the word is out of vocabulary (OOV),
  ///    the result of lookup will be replaced to unknown_token. If the unknown_token is not specified or it is OOV,
  ///    runtime error will be thrown (default={}, means no unknown_token is specified).
  /// \param[in] data_type mindspore::DataType of the tensor after lookup; must be numeric, including bool.
  ///   (default=mindspore::DataType::kNumberTypeInt32).
  Lookup(const std::shared_ptr<Vocab> &vocab, const std::optional<std::vector<char>> &unknown_token,
         mindspore::DataType data_type = mindspore::DataType::kNumberTypeInt32);

  /// \brief Destructor
  ~Lookup() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Generate n-gram from a 1-D string Tensor.
class Ngram final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] ngrams ngrams is a vector of positive integers. For example, if ngrams={4, 3}, then the result
  ///    would be a 4-gram followed by a 3-gram in the same tensor. If the number of words is not enough to make up
  ///    a n-gram, an empty string will be returned.
  /// \param[in] left_pad {"pad_token", pad_width}. Padding performed on left side of the sequence. pad_width will
  ///    be capped at n-1. left_pad=("_",2) would pad the left side of the sequence with "__" (default={"", 0}}).
  /// \param[in] right_pad {"pad_token", pad_width}. Padding performed on right side of the sequence.pad_width will
  ///    be capped at n-1. right_pad=("-",2) would pad the right side of the sequence with "--" (default={"", 0}}).
  /// \param[in] separator Symbol used to join strings together (default=" ").
  explicit Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::string, int32_t> &left_pad = {"", 0},
                 const std::pair<std::string, int32_t> &right_pad = {"", 0}, const std::string &separator = " ")
      : Ngram(ngrams, PairStringToChar(left_pad), PairStringToChar(right_pad), StringToChar(separator)) {}

  /// \brief Constructor.
  /// \param[in] ngrams ngrams is a vector of positive integers. For example, if ngrams={4, 3}, then the result
  ///    would be a 4-gram followed by a 3-gram in the same tensor. If the number of words is not enough to make up
  ///    a n-gram, an empty string will be returned.
  /// \param[in] left_pad {"pad_token", pad_width}. Padding performed on left side of the sequence. pad_width will
  ///    be capped at n-1. left_pad=("_",2) would pad the left side of the sequence with "__" (default={"", 0}}).
  /// \param[in] right_pad {"pad_token", pad_width}. Padding performed on right side of the sequence.pad_width will
  ///    be capped at n-1. right_pad=("-",2) would pad the right side of the sequence with "--" (default={"", 0}}).
  /// \param[in] separator Symbol used to join strings together (default=" ").
  Ngram(const std::vector<int32_t> &ngrams, const std::pair<std::vector<char>, int32_t> &left_pad,
        const std::pair<std::vector<char>, int32_t> &right_pad, const std::vector<char> &separator);

  /// \brief Destructor
  ~Ngram() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

#ifndef _WIN32
/// \brief Apply normalize operation to UTF-8 string tensors.
class NormalizeUTF8 final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] normalize_form Valid values can be any of [NormalizeForm::kNone,NormalizeForm::kNfc,
  ///   NormalizeForm::kNfkc, NormalizeForm::kNfd, NormalizeForm::kNfkd](default=NormalizeForm::kNfkc).
  ///   See http://unicode.org/reports/tr15/ for details.
  ///   - NormalizeForm.kNone, remain the input string tensor unchanged.
  ///   - NormalizeForm.kNfc, normalizes with Normalization Form C.
  ///   - NormalizeForm.kNfkc, normalizes with Normalization Form KC.
  ///   - NormalizeForm.kNfd, normalizes with Normalization Form D.
  ///   - NormalizeForm.kNfkd, normalizes with Normalization Form KD.
  explicit NormalizeUTF8(NormalizeForm normalize_form = NormalizeForm::kNfkc);

  /// \brief Destructor
  ~NormalizeUTF8() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Replace a UTF-8 string tensor with 'replace' according to regular expression 'pattern'.
class RegexReplace final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] pattern The regex expression patterns.
  /// \param[in] replace The string to replace the matched element.
  /// \param[in] replace_all Confirm whether to replace all. If false, only replace the first matched element;
  ///   if true, replace all matched elements (default=true).
  RegexReplace(std::string pattern, std::string replace, bool replace_all = true)
      : RegexReplace(StringToChar(pattern), StringToChar(replace), replace_all) {}

  /// \brief Constructor.
  /// \param[in] pattern The regex expression patterns. Type should be char of vector.
  /// \param[in] replace The string to replace the matched element.
  /// \param[in] replace_all Confirm whether to replace all. If false, only replace the first matched element;
  ///   if true, replace all matched elements (default=true).
  RegexReplace(const std::vector<char> &pattern, const std::vector<char> &replace, bool replace_all);

  /// \brief Destructor
  ~RegexReplace() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string by the regex expression pattern.
class RegexTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] delim_pattern The pattern of regex delimiters.
  /// \param[in] keep_delim_pattern The string matched with 'delim_pattern' can be kept as a token if it can be
  ///   matched by 'keep_delim_pattern'. The default value is an empty string ("").
  ///   which means that delimiters will not be kept as an output token (default="").
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  explicit RegexTokenizer(std::string delim_pattern, std::string keep_delim_pattern = "", bool with_offsets = false)
      : RegexTokenizer(StringToChar(delim_pattern), StringToChar(keep_delim_pattern), with_offsets) {}

  explicit RegexTokenizer(const std::vector<char> &delim_pattern, const std::vector<char> &keep_delim_pattern,
                          bool with_offsets);

  /// \brief Destructor
  ~RegexTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
#endif

/// \brief Tokenize a scalar token or a 1-D token to tokens by sentencepiece.
class SentencePieceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a SentencePieceVocab object.
  /// \param[in] out_type The type of the output.
  SentencePieceTokenizer(const std::shared_ptr<SentencePieceVocab> &vocab,
                         mindspore::dataset::SPieceTokenizerOutType out_type);

  /// \brief Constructor.
  /// \param[in] vocab_path vocab model file path.
  /// \param[in] out_type The type of the output.
  SentencePieceTokenizer(const std::string &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type)
      : SentencePieceTokenizer(StringToChar(vocab_path), out_type) {}

  /// \brief Constructor.
  /// \param[in] vocab_path vocab model file path. type should be char of vector.
  /// \param[in] out_type The type of the output.
  SentencePieceTokenizer(const std::vector<char> &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type);

  /// \brief Destructor
  ~SentencePieceTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Construct a tensor from data (only 1-D for now), where each element in the dimension
///   axis is a slice of data starting at the corresponding position, with a specified width.
class SlidingWindow final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] width The width of the window. It must be an integer and greater than zero.
  /// \param[in] axis The axis where the sliding window is computed (default=0), axis only
  ///    supports 0 or -1 for now.
  explicit SlidingWindow(const int32_t width, const int32_t axis = 0);

  /// \brief Destructor
  ~SlidingWindow() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Convert every element in a string tensor to a number.
///   Strings are cast according to the rules specified in the following links:
///   https://en.cppreference.com/w/cpp/string/basic_string/stof,
///   https://en.cppreference.com/w/cpp/string/basic_string/stoul,
///   except that any strings which represent negative numbers cannot be cast to an unsigned integer type.
class ToNumber final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore::DataType of the tensor to be cast to. Must be a numeric type, excluding bool.
  explicit ToNumber(mindspore::DataType data_type);

  /// \brief Destructor
  ~ToNumber() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
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
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string to Unicode characters.
class UnicodeCharTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  explicit UnicodeCharTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeCharTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize scalar token or 1-D tokens to 1-D sub-word tokens.
class WordpieceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab A Vocab object.
  /// \param[in] suffix_indicator This parameter is used to show that the sub-word
  ///    is the last part of a word (default='##').
  /// \param[in] max_bytes_per_token Tokens exceeding this length will not be further split (default=100).
  /// \param[in] unknown_token When a token cannot be found, return the token directly if 'unknown_token' is an empty
  ///    string, else return the specified string (default='[UNK]').
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  explicit WordpieceTokenizer(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = "##",
                              int32_t max_bytes_per_token = 100, const std::string &unknown_token = "[UNK]",
                              bool with_offsets = false)
      : WordpieceTokenizer(vocab, StringToChar(suffix_indicator), max_bytes_per_token, StringToChar(unknown_token),
                           with_offsets) {}

  explicit WordpieceTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                              int32_t max_bytes_per_token, const std::vector<char> &unknown_token, bool with_offsets);

  /// \brief Destructor
  ~WordpieceTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
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
  /// \param[in] keep_whitespace whether to emit whitespace tokens (default=false).
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  explicit UnicodeScriptTokenizer(bool keep_whitespace = false, bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeScriptTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string on ICU4C defined whitespaces.
class WhitespaceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  explicit WhitespaceTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~WhitespaceTokenizer() = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};
#endif
}  // namespace text
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_INCLUDE_DATASET_TEXT_H_
