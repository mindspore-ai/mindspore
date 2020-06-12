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

#ifndef DATASET_TEXT_VOCAB_H_
#define DATASET_TEXT_VOCAB_H_

#include <string>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dataset/util/status.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace mindspore {
namespace dataset {
namespace py = pybind11;

using WordIdType = int32_t;
using WordType = std::string;

class Vocab {
 public:
  // Build a vocab from a python dictionary key is each word ,id needs to start from 2, no duplicate and continuous
  // @param const py::dict &words - a dictionary containing word, word id pair.
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromPyDict(const py::dict &words, std::shared_ptr<Vocab> *vocab);

  // Build a vocab from a python list, id will be assigned automatically, start from 2
  // @param const py::list &words - a list of string, used to build vocab, id starts from 2
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromPyList(const py::list &words, std::shared_ptr<Vocab> *vocab);

  // Build a vocab from reading a vocab file, id are automatically assigned, start from 2
  // @param std::string &path - path to vocab file , each line is assumed to contain 1 word
  // @param std::string &delimiter - delimiter to break each line with
  // @param int32_t vocab_size - number of words to read from file
  // @param std::shared_ptr<Vocab> *vocab - return value, vocab object
  // @return error code
  static Status BuildFromFile(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                              std::shared_ptr<Vocab> *vocab);

  // Lookup the id of a word, if word doesn't exist in vocab, return default_id
  // @param const WordType word - word to look up
  // @param WordIdType default_id - word id to return to user when its not in the vocab
  // @return WordIdType, word_id
  WordIdType Lookup(const WordType &word, WordIdType default_id) const;

  // reverse lookup, lookup the word based on its id
  // @param WordIdType id - word id to lookup to
  // @return WordType the word
  WordType Lookup(WordIdType id);

  // constructor, shouldn't be called directly, can't be private due to std::make_unique()
  // @param std::unordered_map<WordType, WordIdType> map - sanitized word2id map
  explicit Vocab(std::unordered_map<WordType, WordIdType> map);

  Vocab() = default;

  // add one word to vocab, increment it's index automatically
  // @param std::string & word - word to be added will skip if word already exists
  void append_word(const std::string &word);

  // destructor
  ~Vocab() = default;

  // enum type that holds all special tokens, add more if needed
  enum kSpecialTokens : WordIdType { pad = 0, unk = 1, num_tokens = 2 };

  // reversed lookup table for the reserved tokens
  static const std::vector<WordType> reserved_token_str_;

 private:
  std::unordered_map<WordType, WordIdType> word2id_;
  std::vector<WordType> id2word_;  // reverse lookup
};

}  // namespace dataset
}  // namespace mindspore

#endif  // DATASET_TEXT_VOCAB_H_
