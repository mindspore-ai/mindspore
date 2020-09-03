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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_VOCAB_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_VOCAB_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

#include "minddata/dataset/util/status.h"
#ifdef ENABLE_PYTHON
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#endif

namespace mindspore {
namespace dataset {
#ifdef ENABLE_PYTHON
namespace py = pybind11;
#endif

using WordIdType = int32_t;
using WordType = std::string;

class Vocab {
 public:
#ifdef ENABLE_PYTHON
  // Build a vocab from a python dictionary key is each word ,id needs to start from 2, no duplicate and continuous
  // @param const py::dict &words - a dictionary containing word, word id pair.
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromPyDict(const py::dict &words, std::shared_ptr<Vocab> *vocab);

  // Build a vocab from a python list, id will be assigned automatically, start from 2
  // @param const py::list &words - a list of string, used to build vocab, id starts from 2
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromPyList(const py::list &words, const py::list &special_tokens, bool prepend_special,
                                std::shared_ptr<Vocab> *vocab);

  // Build a vocab from reading a vocab file, id are automatically assigned, start from 2
  // @param std::string &path - path to vocab file , each line is assumed to contain 1 word
  // @param std::string &delimiter - delimiter to break each line with
  // @param int32_t vocab_size - number of words to read from file
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromFile(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                              const py::list &special_tokens, bool prepend_special, std::shared_ptr<Vocab> *vocab);
#endif

  /// \brief Build a vocab from a c++ map. id needs to start from 2, no duplicate and continuous
  /// \param[in] words An unordered_map containing word, word id pair.
  /// \param[out] vocab A vocab object
  /// \return Error code
  static Status BuildFromUnorderedMap(const std::unordered_map<WordType, WordIdType> &words,
                                      std::shared_ptr<Vocab> *vocab);

  /// \brief Build a vocab from a c++ vector. id needs to start from 2, no duplicate and continuous
  /// \param[in] words A vector of string, used to build vocab, id starts from 2
  /// \param[in] special_tokens A vector of string contain special tokens
  /// \param[in] prepend_special Whether special_tokens will be prepended/appended to vocab
  /// \param[out] vocab A vocab object
  /// \return Error code
  static Status BuildFromVector(const std::vector<WordType> &words, const std::vector<WordType> &special_tokens,
                                bool prepend_special, std::shared_ptr<Vocab> *vocab);

  /// \brief Build a vocab from reading a vocab file, id are automatically assigned, start from 2
  /// \param[in] path Path to vocab file , each line is assumed to contain 1 word
  /// \param[in] delimiter Delimiter to break each line with
  /// \param[in] vocab_size Number of words to read from file
  /// \param[in] special_tokens A vector of string contain special tokens
  /// \param[in] prepend_special Whether special_tokens will be prepended/appended to vocab
  /// \param[out] vocab A vocab object
  /// \return Error code
  static Status BuildFromFileCpp(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                                 const std::vector<WordType> &special_tokens, bool prepend_special,
                                 std::shared_ptr<Vocab> *vocab);

  // Lookup the id of a word, if word doesn't exist in vocab, return default_id
  // @param const WordType word - word to look up
  // @param WordIdType default_id - word id to return to user when its not in the vocab
  // @return WordIdType, word_id
  WordIdType Lookup(const WordType &word) const;

  // constructor, shouldn't be called directly, can't be private due to std::make_unique()
  // @param std::unordered_map<WordType, WordIdType> map - sanitized word2id map
  explicit Vocab(std::unordered_map<WordType, WordIdType> map);

  Vocab() = default;

  // add one word to vocab, increment it's index automatically
  // @param std::string & word - word to be added will skip if word already exists
  void append_word(const std::string &word);

  // return a read-only vocab
  const std::unordered_map<WordType, WordIdType> vocab() { return word2id_; }

  // destructor
  ~Vocab() = default;

  static const WordIdType kNoTokenExists;

 private:
  std::unordered_map<WordType, WordIdType> word2id_;
};

}  // namespace dataset
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_VOCAB_H_
