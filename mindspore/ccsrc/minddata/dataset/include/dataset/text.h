/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <utility>
#include <vector>

#include "include/api/dual_abi_helper.h"
#include "include/api/status.h"
#include "include/dataset/constants.h"
#include "include/dataset/transforms.h"

namespace mindspore {
namespace dataset {
class TensorOperation;
class Vectors;

using WordIdType = int32_t;
using WordType = std::string;

/// \brief Vocab object that is used to save pairs of words and ids.
/// \note It contains a map that maps each word(str) to an id(int) or reverse.
class Vocab {
 public:
  /// \brief Build a vocab from an unordered_map. IDs should be no duplicate and continuous.
  /// \param[in] words An unordered_map containing word id pair.
  /// \param[out] vocab A vocab object.
  /// \return Status code.
  /// \par Example
  /// \code
  ///     // Build a map
  ///     std::unordered_map<std::string, int32_t> dict;
  ///     dict["banana"] = 0;
  ///     dict["apple"] = 1;
  ///     dict["cat"] = 2;
  ///     dict["dog"] = 3;
  ///     // Build vocab from map
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromUnorderedMap(dict, &vocab);
  /// \endcode
  static Status BuildFromUnorderedMap(const std::unordered_map<WordType, WordIdType> &words,
                                      std::shared_ptr<Vocab> *vocab);

  /// \brief Build a vocab from a c++ vector. id no duplicate and continuous.
  /// \param[in] words A vector of string containing words.
  /// \param[in] special_tokens A vector of string containing special tokens.
  /// \param[in] prepend_special Whether the special_tokens will be prepended/appended to vocab.
  /// \param[out] vocab A vocab object.
  /// \return Status code.
  /// \par Example
  /// \code
  ///     // Build vocab from a vector of words, special tokens are prepended to vocab
  ///     std::vector<std::string> list = {"apple", "banana", "cat", "dog", "egg"};
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromVector(list, {"<unk>"}, true, &vocab);
  /// \endcode
  static Status BuildFromVector(const std::vector<WordType> &words, const std::vector<WordType> &special_tokens,
                                bool prepend_special, std::shared_ptr<Vocab> *vocab);

  /// \brief Build a vocab from vocab file, IDs will be automatically assigned.
  /// \param[in] path Path to vocab file, each line in file is assumed as a word (including space).
  /// \param[in] delimiter Delimiter to break each line, characters after the delimiter will be deprecated.
  /// \param[in] vocab_size Number of lines to be read from file.
  /// \param[in] special_tokens A vector of string containing special tokens.
  /// \param[in] prepend_special Whether the special_tokens will be prepended/appended to vocab.
  /// \param[out] vocab A vocab object.
  /// \return Status code.
  /// \par Example
  /// \code
  ///     // Build vocab from local file
  ///     std::string vocab_dir = datasets_root_path_ + "/testVocab/vocab_list.txt";
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromFile(vocab_dir, ",", -1, {"<pad>", "<unk>"}, true, &vocab);
  /// \endcode
  static Status BuildFromFile(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                              const std::vector<WordType> &special_tokens, bool prepend_special,
                              std::shared_ptr<Vocab> *vocab);

  /// Lookup the id of a word, if the word doesn't exist in vocab, return -1.
  /// \param word Word to be looked up.
  /// \return ID of the word in the vocab.
  /// \par Example
  /// \code
  ///     // lookup, convert token to id
  ///     auto single_index = vocab->TokensToIds("home");
  ///     single_index = vocab->TokensToIds("hello");
  /// \endcode
  WordIdType TokensToIds(const WordType &word) const;

  /// Lookup the id of a word, if the word doesn't exist in vocab, return -1.
  /// \param words Words to be looked up.
  /// \return ID of the word in the vocab.
  /// \par Example
  /// \code
  ///     // lookup multiple tokens
  ///     auto multi_indexs = vocab->TokensToIds(std::vector<std::string>{"<pad>", "behind"});
  ///     std::vector<int32_t> expected_multi_indexs = {0, 4};
  ///     multi_indexs = vocab->TokensToIds(std::vector<std::string>{"<pad>", "apple"});
  ///     expected_multi_indexs = {0, -1};
  /// \endcode
  std::vector<WordIdType> TokensToIds(const std::vector<WordType> &words) const;

  /// Lookup the word of an ID, if ID doesn't exist in vocab, return empty string.
  /// \param id ID to be looked up.
  /// \return Indicates the word corresponding to the ID.
  /// \par Example
  /// \code
  ///     // reverse lookup, convert id to token
  ///     auto single_word = vocab->IdsToTokens(2);
  ///     single_word = vocab->IdsToTokens(-1);
  /// \endcode
  WordType IdsToTokens(const WordIdType &id);

  /// Lookup the word of an ID, if ID doesn't exist in vocab, return empty string.
  /// \param ids ID to be looked up.
  /// \return Indicates the word corresponding to the ID.
  /// \par Example
  /// \code
  ///     // reverse lookup multiple ids
  ///     auto multi_words = vocab->IdsToTokens(std::vector<int32_t>{0, 4});
  ///     std::vector<std::string> expected_multi_words = {"<pad>", "behind"};
  ///     multi_words = vocab->IdsToTokens(std::vector<int32_t>{0, 99});
  ///     expected_multi_words = {"<pad>", ""};
  /// \endcode
  std::vector<WordType> IdsToTokens(const std::vector<WordIdType> &ids);

  /// Constructor, shouldn't be called directly, can't be private due to std::make_unique().
  /// \param map Sanitized word2id map.
  explicit Vocab(std::unordered_map<WordType, WordIdType> map);

  /// \brief Add one word to vocab, increment it's index automatically.
  /// \param word Word to be added, word will skip if word already exists.
  void AppendWord(const std::string &word);

  /// \brief Return a read-only vocab in unordered_map type.
  /// \return A unordered_map of word2id.
  const std::unordered_map<WordType, WordIdType> &GetVocab() const { return word2id_; }

  /// \brief Constructor.
  Vocab() = default;

  /// \brief Destructor.
  ~Vocab() = default;

  static const WordIdType kNoTokenExists;
  static const WordType kNoIdExists;

 private:
  std::unordered_map<WordType, WordIdType> word2id_;
  std::unordered_map<WordIdType, WordType> id2word_;
};

/// \brief SentencePiece object that is used to do words segmentation.
class SentencePieceVocab {
 public:
  /// \brief Build a SentencePiece object from a file.
  /// \param[in] path_list Path to the file which contains the SentencePiece list.
  /// \param[in] vocab_size Vocabulary size.
  /// \param[in] character_coverage Amount of characters covered by the model, good defaults are: 0.9995 for
  ///              languages with rich character set like Japanese or Chinese and 1.0 for other languages with small
  ///              character set.
  /// \param[in] model_type It can be any of [SentencePieceModel::kUnigram, SentencePieceModel::kBpe,
  ///              SentencePieceModel::kChar, SentencePieceModel::kWord], default is SentencePieceModel::kUnigram. The
  ///              input sentence must be pre-tokenized when using SentencePieceModel.WORD type.
  ///              - SentencePieceModel.kUnigram, Unigram Language Model means the next word in the sentence is assumed
  ///                to be independent of the previous words generated by the model.
  ///              - SentencePieceModel.kBpe, refers to byte pair encoding algorithm, which replaces the most frequent
  ///                pair of bytes in a sentence with a single, unused byte.
  ///              - SentencePieceModel.kChar, refers to char based sentencePiece Model type.
  ///              - SentencePieceModel.kWord, refers to word based sentencePiece Model type.
  /// \param[in] params A dictionary with no incoming parameters(The parameters are derived from SentencePiece library).
  /// \param[out] vocab A SentencePieceVocab object.
  /// \return SentencePieceVocab, vocab built from the file.
  /// \par Example
  /// \code
  ///     std::string dataset_path;
  ///     dataset_path = datasets_root_path_ + "/test_sentencepiece/vocab.txt";
  ///     std::vector<std::string> path_list;
  ///     path_list.emplace_back(dataset_path);
  ///     std::unordered_map<std::string, std::string> param_map;
  ///     std::shared_ptr<SentencePieceVocab> spm = std::make_unique<SentencePieceVocab>();
  ///     Status rc = SentencePieceVocab::BuildFromFile(path_list, 5000, 0.9995,
  ///                                                   SentencePieceModel::kUnigram, param_map, &spm);
  /// \endcode
  static Status BuildFromFile(const std::vector<std::string> &path_list, int32_t vocab_size, float character_coverage,
                              const SentencePieceModel &model_type,
                              const std::unordered_map<std::string, std::string> &params,
                              std::shared_ptr<SentencePieceVocab> *vocab);

  /// \brief Save the SentencePiece model into given file path.
  /// \param[in] vocab A SentencePiece object to be saved.
  /// \param[in] path Path to store the model.
  /// \param[in] filename The save name of model file.
  /// \par Example
  /// \code
  ///     // Save vocab model to local
  ///     vocab->SaveModel(&vocab, datasets_root_path_ + "/test_sentencepiece", "m.model");
  /// \endcode
  static Status SaveModel(const std::shared_ptr<SentencePieceVocab> *vocab, const std::string &path,
                          const std::string &filename);

  /// \brief Constructor.
  SentencePieceVocab();

  /// \brief Destructor.
  ~SentencePieceVocab() = default;

  const std::string &model_proto();

  void set_model_proto(const std::string &model_proto);

 private:
  std::string model_proto_;
};

// Transform operations for text
namespace text {
/// \brief Add token to beginning or end of sequence.
class DATASET_API AddToken final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] token The token to be added.
  /// \param[in] begin Whether to insert token at start or end of sequence. Default: true.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto add_token_op = text::AddToken(token='TOKEN', begin=True);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({add_token_op},   // operations
  ///                            {"text"});       // input columns
  /// \endcode
  explicit AddToken(const std::string &token, bool begin = true);

  /// \brief Destructor.
  ~AddToken() override = default;

 protected:
  /// \brief Function to convert TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

#ifndef _WIN32
/// \brief Tokenize a scalar tensor of UTF-8 string by specific rules.
/// \note BasicTokenizer is not supported on the Windows platform yet.
class DATASET_API BasicTokenizer final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::BasicTokenizer();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit BasicTokenizer(bool lower_case = false, bool keep_whitespace = false,
                          NormalizeForm normalize_form = NormalizeForm::kNone, bool preserve_unused_token = true,
                          bool with_offsets = false);

  /// \brief Destructor
  ~BasicTokenizer() override = default;

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
class DATASET_API BertTokenizer final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::vector<std::string> list = {"a", "b", "c", "d"};
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  ///     auto tokenizer_op = text::BertTokenizer(vocab);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
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
                bool keep_whitespace, NormalizeForm normalize_form, bool preserve_unused_token, bool with_offsets);

  /// \brief Destructor
  ~BertTokenizer() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Apply case fold operation on UTF-8 string tensors.
class DATASET_API CaseFold final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto casefold_op = text::CaseFold();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({casefold_op},   // operations
  ///                            {"text"});       // input columns
  /// \endcode
  CaseFold();

  /// \brief Destructor
  ~CaseFold() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};

/// \brief Filter wikipedia xml lines.
class FilterWikipediaXML final : public TensorTransform {
 public:
  /// \brief Constructor.
  FilterWikipediaXML();

  /// \brief Destructor
  ~FilterWikipediaXML() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;
};
#endif

/// \brief Tokenize a Chinese string into words based on the dictionary.
/// \note The integrity of the HMMSegment algorithm and MPSegment algorithm files must be confirmed.
class DATASET_API JiebaTokenizer final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::JiebaTokenizer("/path/to/hmm/file", "/path/to/mp/file");
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
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
  ~JiebaTokenizer() override = default;

  /// \brief Add a user defined word to the JiebaTokenizer's dictionary.
  /// \param[in] word The word to be added to the JiebaTokenizer instance.
  ///   The added word will not be written into the built-in dictionary on disk.
  /// \param[in] freq The frequency of the word to be added. The higher the frequency,
  ///   the better chance the word will be tokenized (default=None, use default frequency).
  /// \return Status error code, returns OK if no error is encountered.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::JiebaTokenizer("/path/to/hmm/file", "/path/to/mp/file");
  ///
  ///     Status s = tokenizer_op.AddWord("hello", 2);
  /// \endcode
  Status AddWord(const std::string &word, int64_t freq = 0) { return AddWordChar(StringToChar(word), freq); }

  /// \brief Add a user defined dictionary of word-freq pairs to the JiebaTokenizer's dictionary.
  /// \param[in] user_dict Vector of word-freq pairs to be added to the JiebaTokenizer's dictionary.
  /// \return Status error code, returns OK if no error is encountered.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::JiebaTokenizer("/path/to/hmm/file", "/path/to/mp/file");
  ///
  ///     std::vector<std::pair<std::string, int64_t>> user_dict = {{"a", 1}, {"b", 2}, {"c", 3}};
  ///     Status s = tokenizer_op.AddDict(user_dict);
  /// \endcode
  Status AddDict(const std::vector<std::pair<std::string, int64_t>> &user_dict) {
    return AddDictChar(PairStringInt64ToPairCharInt64(user_dict));
  }

  /// \brief Add user defined dictionary of word-freq pairs to the JiebaTokenizer's dictionary from a file.
  ///   Only valid word-freq pairs in user defined file will be added into the dictionary.
  ///   Rows containing invalid inputs will be ignored, no error nor warning status is returned.
  /// \param[in] file_path Path to the dictionary which includes user defined word-freq pairs.
  /// \return Status error code, returns OK if no error is encountered.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::JiebaTokenizer("/path/to/hmm/file", "/path/to/mp/file");
  ///
  ///     Status s = tokenizer_op.AddDict("/path/to/dict/file");
  /// \endcode
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
class DATASET_API Lookup final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a Vocab object.
  /// \param[in] unknown_token Word is used for lookup. In case of the word is out of vocabulary (OOV),
  ///    the result of lookup will be replaced to unknown_token. If the unknown_token is not specified or it is OOV,
  ///    runtime error will be thrown (default={}, means no unknown_token is specified).
  /// \param[in] data_type mindspore::DataType of the tensor after lookup; must be numeric, including bool.
  ///   (default=mindspore::DataType::kNumberTypeInt32).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///    std::vector<std::string> list = {"a", "b", "c", "d"};
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromVector(list, {}, true, &vocab);
  ///     auto lookup_op = text::Lookup(vocab, "[unk]");
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({lookup_op},   // operations
  ///                            {"text"});     // input columns
  /// \endcode
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
  ~Lookup() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Generate n-gram from a 1-D string Tensor.
class DATASET_API Ngram final : public TensorTransform {
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
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto ngram_op = text::Ngram({2, 3}, {"&", 2}, {"&", 2}, "-");
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({ngram_op},   // operations
  ///                            {"text"});    // input columns
  /// \endcode
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
  ~Ngram() override = default;

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
class DATASET_API NormalizeUTF8 final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] normalize_form Valid values can be any of [NormalizeForm::kNone,NormalizeForm::kNfc,
  ///   NormalizeForm::kNfkc, NormalizeForm::kNfd, NormalizeForm::kNfkd](default=NormalizeForm::kNfkc).
  ///   See <http://unicode.org/reports/tr15/> for details.
  ///   - NormalizeForm.kNone, remain the input string tensor unchanged.
  ///   - NormalizeForm.kNfc, normalizes with Normalization Form C.
  ///   - NormalizeForm.kNfkc, normalizes with Normalization Form KC.
  ///   - NormalizeForm.kNfd, normalizes with Normalization Form D.
  ///   - NormalizeForm.kNfkd, normalizes with Normalization Form KD.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto normalizeutf8_op = text::NormalizeUTF8();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({normalizeutf8_op},   // operations
  ///                            {"text"});            // input columns
  /// \endcode
  explicit NormalizeUTF8(NormalizeForm normalize_form = NormalizeForm::kNfkc);

  /// \brief Destructor
  ~NormalizeUTF8() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Replace a UTF-8 string tensor with 'replace' according to regular expression 'pattern'.
class DATASET_API RegexReplace final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] pattern The regex expression patterns.
  /// \param[in] replace The string to replace the matched element.
  /// \param[in] replace_all Confirm whether to replace all. If false, only replace the first matched element;
  ///   if true, replace all matched elements (default=true).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto regex_op = text::RegexReplace("\\s+", "_", true);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({regex_op},   // operations
  ///                            {"text"});    // input columns
  /// \endcode
  RegexReplace(const std::string &pattern, const std::string &replace, bool replace_all = true)
      : RegexReplace(StringToChar(pattern), StringToChar(replace), replace_all) {}

  /// \brief Constructor.
  /// \param[in] pattern The regex expression patterns. Type should be char of vector.
  /// \param[in] replace The string to replace the matched element.
  /// \param[in] replace_all Confirm whether to replace all. If false, only replace the first matched element;
  ///   if true, replace all matched elements (default=true).
  RegexReplace(const std::vector<char> &pattern, const std::vector<char> &replace, bool replace_all);

  /// \brief Destructor
  ~RegexReplace() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string by the regex expression pattern.
class DATASET_API RegexTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] delim_pattern The pattern of regex delimiters.
  /// \param[in] keep_delim_pattern The string matched with 'delim_pattern' can be kept as a token if it can be
  ///   matched by 'keep_delim_pattern'. The default value is an empty string ("").
  ///   which means that delimiters will not be kept as an output token (default="").
  /// \param[in] with_offsets Whether to output offsets of tokens (default=false).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto regex_op = text::RegexTokenizer("\\s+", "\\s+", false);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({regex_op},   // operations
  ///                            {"text"});    // input columns
  /// \endcode
  explicit RegexTokenizer(const std::string &delim_pattern, const std::string &keep_delim_pattern = "",
                          bool with_offsets = false)
      : RegexTokenizer(StringToChar(delim_pattern), StringToChar(keep_delim_pattern), with_offsets) {}

  explicit RegexTokenizer(const std::vector<char> &delim_pattern, const std::vector<char> &keep_delim_pattern,
                          bool with_offsets);

  /// \brief Destructor
  ~RegexTokenizer() override = default;

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
class DATASET_API SentencePieceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab a SentencePieceVocab object.
  /// \param[in] out_type The type of the output.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::shared_ptr<Dataset> ds_vocab = TextFile({"/path/to/vocab/file"}, 0, ShuffleMode::kFalse);
  ///     std::shared_ptr<SentencePieceVocab> vocab =
  ///         ds_vocab->BuildSentencePieceVocab({}, 0, 0.9995, SentencePieceModel::kUnigram, {});
  ///     auto tokenizer_op = text::SentencePieceTokenizer(vocab, mindspore::dataset::SPieceTokenizerOutType::kString);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  SentencePieceTokenizer(const std::shared_ptr<SentencePieceVocab> &vocab,
                         mindspore::dataset::SPieceTokenizerOutType out_type);

  /// \brief Constructor.
  /// \param[in] vocab_path vocab model file path.
  /// \param[in] out_type The type of the output.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::SentencePieceTokenizer("/path/to/model",
  ///                                                      mindspore::dataset::SPieceTokenizerOutType::kInt);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  SentencePieceTokenizer(const std::string &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type)
      : SentencePieceTokenizer(StringToChar(vocab_path), out_type) {}

  /// \brief Constructor.
  /// \param[in] vocab_path vocab model file path. type should be char of vector.
  /// \param[in] out_type The type of the output.
  SentencePieceTokenizer(const std::vector<char> &vocab_path, mindspore::dataset::SPieceTokenizerOutType out_type);

  /// \brief Destructor
  ~SentencePieceTokenizer() override = default;

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
class DATASET_API SlidingWindow final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] width The width of the window. It must be an integer and greater than zero.
  /// \param[in] axis The axis where the sliding window is computed (default=0), axis only
  ///    supports 0 or -1 for now.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto slidingwindow_op = text::SlidingWindow(5, 0);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({slidingwindow_op},   // operations
  ///                            {"text"});            // input columns
  /// \endcode
  explicit SlidingWindow(int32_t width, int32_t axis = 0);

  /// \brief Destructor
  ~SlidingWindow() override = default;

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
class DATASET_API ToNumber final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] data_type mindspore::DataType of the tensor to be cast to. Must be a numeric type, excluding bool.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto to_number_op = text::ToNumber(mindspore::DataType::kNumberTypeInt8);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({to_number_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit ToNumber(mindspore::DataType data_type);

  /// \brief Destructor
  ~ToNumber() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Look up a token into an vector according to the input Vectors table.
class DATASET_API ToVectors final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vectors A Vectors object.
  /// \param[in] unk_init In case of the token is out-of-vectors (OOV), the result will be initialized with `unk_init`.
  ///     (default={}, means to initialize with zero vectors).
  /// \param[in] lower_case_backup Whether to look up the token in the lower case (default=false).
  explicit ToVectors(const std::shared_ptr<Vectors> &vectors, const std::vector<float> &unk_init = {},
                     bool lower_case_backup = false);

  /// \brief Destructor
  ~ToVectors() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Truncate the input sequence so that it does not exceed the maximum length.
class DATASET_API Truncate final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] max_seq_len Maximum allowable length.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto truncate_op = text::Truncate(5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({truncate_op},   // operations
  ///                            {"text"});       // input columns
  /// \endcode
  explicit Truncate(int32_t max_seq_len);

  /// \brief Destructor.
  ~Truncate() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Truncate a pair of rank-1 tensors such that the total length is less than max_length.
class DATASET_API TruncateSequencePair final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] max_length Maximum length required.
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto truncate_op = text::TruncateSequencePair(5);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({truncate_op},   // operations
  ///                            {"text"});       // input columns
  /// \endcode
  explicit TruncateSequencePair(int32_t max_length);

  /// \brief Destructor
  ~TruncateSequencePair() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string to Unicode characters.
class DATASET_API UnicodeCharTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::UnicodeCharTokenizer();
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit UnicodeCharTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeCharTokenizer() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize scalar token or 1-D tokens to 1-D sub-word tokens.
class DATASET_API WordpieceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] vocab A Vocab object.
  /// \param[in] suffix_indicator This parameter is used to show that the sub-word
  ///    is the last part of a word (default='##').
  /// \param[in] max_bytes_per_token Tokens exceeding this length will not be further split (default=100).
  /// \param[in] unknown_token When a token cannot be found, return the token directly if 'unknown_token' is an empty
  ///    string, else return the specified string (default='[UNK]').
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     std::vector<std::string> word_list = {"book", "apple", "rabbit"};
  ///     std::shared_ptr<Vocab> vocab = std::make_shared<Vocab>();
  ///     Status s = Vocab::BuildFromVector(word_list, {}, true, &vocab);
  ///     auto tokenizer_op = text::WordpieceTokenizer(vocab);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit WordpieceTokenizer(const std::shared_ptr<Vocab> &vocab, const std::string &suffix_indicator = "##",
                              int32_t max_bytes_per_token = 100, const std::string &unknown_token = "[UNK]",
                              bool with_offsets = false)
      : WordpieceTokenizer(vocab, StringToChar(suffix_indicator), max_bytes_per_token, StringToChar(unknown_token),
                           with_offsets) {}

  explicit WordpieceTokenizer(const std::shared_ptr<Vocab> &vocab, const std::vector<char> &suffix_indicator,
                              int32_t max_bytes_per_token, const std::vector<char> &unknown_token, bool with_offsets);

  /// \brief Destructor
  ~WordpieceTokenizer() override = default;

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
class DATASET_API UnicodeScriptTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] keep_whitespace whether to emit whitespace tokens (default=false).
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::UnicodeScriptTokenizer(false, true);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit UnicodeScriptTokenizer(bool keep_whitespace = false, bool with_offsets = false);

  /// \brief Destructor
  ~UnicodeScriptTokenizer() override = default;

 protected:
  /// \brief The function to convert a TensorTransform object into a TensorOperation object.
  /// \return Shared pointer to the TensorOperation object.
  std::shared_ptr<TensorOperation> Parse() override;

 private:
  struct Data;
  std::shared_ptr<Data> data_;
};

/// \brief Tokenize a scalar tensor of UTF-8 string on ICU4C defined whitespaces.
class DATASET_API WhitespaceTokenizer final : public TensorTransform {
 public:
  /// \brief Constructor.
  /// \param[in] with_offsets whether to output offsets of tokens (default=false).
  /// \par Example
  /// \code
  ///     /* Define operations */
  ///     auto tokenizer_op = text::WhitespaceTokenizer(false);
  ///
  ///     /* dataset is an instance of Dataset object */
  ///     dataset = dataset->Map({tokenizer_op},   // operations
  ///                            {"text"});        // input columns
  /// \endcode
  explicit WhitespaceTokenizer(bool with_offsets = false);

  /// \brief Destructor
  ~WhitespaceTokenizer() override = default;

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
