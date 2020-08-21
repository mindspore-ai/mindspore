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
#include <algorithm>

#include "minddata/dataset/text/vocab.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace dataset {
Vocab::Vocab(std::unordered_map<WordType, WordIdType> word2id) { word2id_ = std::move(word2id); }

WordIdType Vocab::Lookup(const WordType &word) const {
  auto itr = word2id_.find(word);
  return itr == word2id_.end() ? kNoTokenExists : itr->second;
}

#ifdef ENABLE_PYTHON
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

Status Vocab::BuildFromPyDict(const py::dict &words, std::shared_ptr<Vocab> *vocab) {
  std::unordered_map<WordType, WordIdType> word2id;
  for (auto p : words) {
    word2id[py::str(p.first)] = py::reinterpret_borrow<py::int_>(p.second);
  }
  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}
#endif

void Vocab::append_word(const std::string &word) {
  if (word2id_.find(word) == word2id_.end()) {
    word2id_[word] = word2id_.size();
  }
}

Status Vocab::BuildFromUnorderedMap(const std::unordered_map<WordType, WordIdType> &words,
                                    std::shared_ptr<Vocab> *vocab) {
  // Validate parameters and build map
  std::unordered_map<WordType, WordIdType> word2id;
  for (auto p : words) {
    if (p.second < 0) {
      MS_LOG(ERROR) << "index can not be negetive, but got " << p.second;
      RETURN_STATUS_UNEXPECTED("index can not be negetive, but got " + std::to_string(p.second));
    }
    word2id[p.first] = p.second;
  }
  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromVector(const std::vector<WordType> &words, const std::vector<WordType> &special_tokens,
                              bool prepend_special, std::shared_ptr<Vocab> *vocab) {
  // Validate parameters
  std::string duplicate_word;
  for (const WordType &word : words) {
    if (std::count(words.begin(), words.end(), word) > 1) {
      if (duplicate_word.find(word) == std::string::npos) {
        duplicate_word = duplicate_word + ", " + word;
      }
    }
  }
  if (!duplicate_word.empty()) {
    MS_LOG(ERROR) << "words contains duplicate word: " << duplicate_word;
    RETURN_STATUS_UNEXPECTED("words contains duplicate word: " + duplicate_word);
  }

  std::string duplicate_sp;
  for (const WordType &sp : special_tokens) {
    if (std::count(special_tokens.begin(), special_tokens.end(), sp) > 1) {
      if (duplicate_sp.find(sp) == std::string::npos) {
        duplicate_sp = duplicate_sp + ", " + sp;
      }
    }
  }
  if (!duplicate_sp.empty()) {
    MS_LOG(ERROR) << "special_tokens contains duplicate word: " << duplicate_sp;
    RETURN_STATUS_UNEXPECTED("special_tokens contains duplicate word: " + duplicate_sp);
  }

  std::unordered_map<WordType, WordIdType> word2id;

  // if special is added in front, normal words id will start from number of special tokens
  WordIdType word_id = prepend_special ? static_cast<WordIdType>(special_tokens.size()) : 0;
  for (auto word : words) {
    word2id[word] = word_id++;
  }

  word_id = prepend_special ? 0 : word2id.size();

  for (auto special_token : special_tokens) {
    word2id[special_token] = word_id++;
  }

  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromFileCpp(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                               const std::vector<WordType> &special_tokens, bool prepend_special,
                               std::shared_ptr<Vocab> *vocab) {
  // Validate parameters
  if (vocab_size < 0 && vocab_size != -1) {
    MS_LOG(ERROR) << "vocab_size shoule be either -1 or positive integer, but got " << vocab_size;
    RETURN_STATUS_UNEXPECTED("vocab_size shoule be either -1 or positive integer, but got " +
                             std::to_string(vocab_size));
  }

  std::string duplicate_sp;
  for (const WordType &sp : special_tokens) {
    if (std::count(special_tokens.begin(), special_tokens.end(), sp) > 1) {
      if (duplicate_sp.find(sp) == std::string::npos) {
        duplicate_sp = duplicate_sp + ", " + sp;
      }
    }
  }
  if (!duplicate_sp.empty()) {
    MS_LOG(ERROR) << "special_tokens contains duplicate word: " << duplicate_sp;
    RETURN_STATUS_UNEXPECTED("special_tokens contains duplicate word: " + duplicate_sp);
  }

  std::unordered_set<std::string> specials;
  // used to check that words in file don't contain any special token that already exists
  for (auto word : special_tokens) {
    specials.insert(word);
  }
  WordIdType word_id = prepend_special ? static_cast<WordIdType>(special_tokens.size()) : 0;
  std::unordered_map<WordType, WordIdType> word2id;
  std::fstream handle(path, std::ios::in);
  if (!handle.good() || !handle.is_open()) {
    MS_LOG(ERROR) << "fail to open:" + path;
    RETURN_STATUS_UNEXPECTED("fail to open:" + path);
  }
  std::string word;
  while (std::getline(handle, word)) {
    if (!delimiter.empty()) {
      // if delimiter is not found, find_first_of would return std::string::npos which is -1
      word = word.substr(0, word.find_first_of(delimiter));
    }
    if (word2id.find(word) != word2id.end()) {
      MS_LOG(ERROR) << "duplicate word:" + word + ".";
      RETURN_STATUS_UNEXPECTED("duplicate word:" + word + ".");
    }
    if (specials.find(word) != specials.end()) {
      MS_LOG(ERROR) << word + " is already in special_tokens.";
      RETURN_STATUS_UNEXPECTED(word + " is already in special_tokens.");
    }
    word2id[word] = word_id++;
    // break if enough row is read, if vocab_size is smaller than 0
    if (word2id.size() == vocab_size) break;
  }

  word_id = prepend_special ? 0 : word2id.size();

  for (auto special_token : special_tokens) {
    word2id[special_token] = word_id++;
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

const WordIdType Vocab::kNoTokenExists = -1;

}  // namespace dataset
}  // namespace mindspore
