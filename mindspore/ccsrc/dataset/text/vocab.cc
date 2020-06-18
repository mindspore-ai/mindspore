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
#include <fstream>
#include <unordered_set>
#include <unordered_map>
#include <utility>

#include "dataset/text/vocab.h"

namespace mindspore {
namespace dataset {
Vocab::Vocab(std::unordered_map<WordType, WordIdType> word2id) { word2id_ = std::move(word2id); }

WordIdType Vocab::Lookup(const WordType &word, WordIdType default_id) const {
  auto itr = word2id_.find(word);
  return itr == word2id_.end() ? default_id : itr->second;
}

Status Vocab::BuildFromPyList(const py::list &words, const py::list &special_tokens, bool prepend_special,
                              std::shared_ptr<Vocab> *vocab) {
  // check of duplication on both words and special_tokens will be performed in python
  // special_tokens and words both need to be unique, and shouldn't overlap
  std::unordered_map<WordType, WordIdType> word2id;
  // if special is added in front, normal words id will start from number of special tokens
  WordIdType word_id = prepend_special ? static_cast<WordIdType>(special_tokens.size()) : 0;

  for (auto word : words) {
    word2id[py::str(word)] = word_id++;
  }

  word_id = prepend_special ? 0 : word2id.size();

  for (auto special_token : special_tokens) {
    word2id[py::str(special_token)] = word_id++;
  }

  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromFile(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                            const py::list &special_tokens, bool prepend_special, std::shared_ptr<Vocab> *vocab) {
  // python validator checks special_tokens doesn't contain any duplicate words
  std::unordered_set<std::string> specials;
  // used to check that words in file don't contain any special token that already exists
  for (auto word : special_tokens) {
    specials.insert(py::str(word));
  }
  WordIdType word_id = prepend_special ? static_cast<WordIdType>(special_tokens.size()) : 0;
  std::unordered_map<WordType, WordIdType> word2id;
  std::fstream handle(path, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(handle.good() && handle.is_open(), "fail to open:" + path);
  std::string word;
  while (std::getline(handle, word)) {
    if (!delimiter.empty()) {
      // if delimiter is not found, find_first_of would return std::string::npos which is -1
      word = word.substr(0, word.find_first_of(delimiter));
    }
    CHECK_FAIL_RETURN_UNEXPECTED(word2id.find(word) == word2id.end(), "duplicate word:" + word + ".");
    CHECK_FAIL_RETURN_UNEXPECTED(specials.find(word) == specials.end(), word + " is already in special_tokens.");
    word2id[word] = word_id++;
    // break if enough row is read, if vocab_size is smaller than 0
    if (word2id.size() == vocab_size) break;
  }

  word_id = prepend_special ? 0 : word2id.size();

  for (auto special_token : special_tokens) {
    word2id[py::str(special_token)] = word_id++;
  }

  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromPyDict(const py::dict &words, std::shared_ptr<Vocab> *vocab) {
  std::unordered_map<WordType, WordIdType> word2id;
  for (auto p : words) {
    word2id[py::str(p.first)] = py::reinterpret_borrow<py::int_>(p.second);
  }
  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

void Vocab::append_word(const std::string &word) {
  if (word2id_.find(word) == word2id_.end()) {
    word2id_[word] = word2id_.size();
  }
}
}  // namespace dataset
}  // namespace mindspore
