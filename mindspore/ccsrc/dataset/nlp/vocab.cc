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
#include <map>
#include <utility>

#include "dataset/nlp/vocab.h"

namespace mindspore {
namespace dataset {
Vocab::Vocab(std::unordered_map<WordType, WordIdType> word2id) {
  word2id_ = std::move(word2id);
  id2word_.resize(word2id_.size());
  for (auto p : word2id_) {
    id2word_[p.second - kSpecialTokens::num_tokens] = p.first;
  }
}

WordIdType Vocab::Lookup(const WordType &word, WordIdType default_id) const {
  auto itr = word2id_.find(word);
  return itr == word2id_.end() ? default_id : itr->second;
}
WordType Vocab::Lookup(WordIdType id) const {
  if (id < kSpecialTokens::num_tokens) {
    return reserved_token_str_[id];
  } else if (id - kSpecialTokens::num_tokens >= id2word_.size()) {
    return reserved_token_str_[kSpecialTokens::unk];
  } else {
    return id2word_[id - kSpecialTokens::num_tokens];
  }
}

Status Vocab::BuildFromPyList(const py::list &words, std::shared_ptr<Vocab> *vocab) {
  std::unordered_map<WordType, WordIdType> word2id;
  WordIdType word_id = kSpecialTokens::num_tokens;
  for (auto word : words) {
    const std::string s = py::str(word);
    CHECK_FAIL_RETURN_UNEXPECTED(word2id.find(s) == word2id.end(), "duplicate word:" + s);
    word2id[s] = word_id++;
  }
  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromFile(const std::string &path, const std::string &delimiter, int32_t vocab_size,
                            std::shared_ptr<Vocab> *vocab) {
  std::unordered_map<WordType, WordIdType> word2id;
  WordIdType word_id = kSpecialTokens::num_tokens;
  std::fstream handle(path, std::ios::in);
  CHECK_FAIL_RETURN_UNEXPECTED(handle.good() && handle.is_open(), "fail to open:" + path);
  std::string word;
  while (std::getline(handle, word)) {
    if (!delimiter.empty()) {
      // if delimiter is not found, find_first_of would return std::string::npos which is -1
      word = word.substr(0, word.find_first_of(delimiter));
    }
    CHECK_FAIL_RETURN_UNEXPECTED(word2id.find(word) == word2id.end(), "duplicate word:" + word);
    word2id[word] = word_id++;
    // break if enough row is read, if vocab_size is smaller than 0
    if (word_id == vocab_size + kSpecialTokens::num_tokens) break;
  }
  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}

Status Vocab::BuildFromPyDict(const py::dict &words, std::shared_ptr<Vocab> *vocab) {
  std::unordered_map<WordType, WordIdType> word2id;
  std::map<WordIdType, WordType> id2word;
  for (auto p : words) {
    WordIdType word_id = py::reinterpret_borrow<py::int_>(p.second);
    if (word_id < kSpecialTokens::num_tokens) continue;  // skip id that are reserved
    std::string word = py::str(p.first);
    CHECK_FAIL_RETURN_UNEXPECTED(id2word.find(word_id) == id2word.end(), "duplicate id:" + word);
    id2word[word_id] = word;
  }

  WordIdType cnt = kSpecialTokens::num_tokens;
  for (auto p : id2word) {
    CHECK_FAIL_RETURN_UNEXPECTED(p.first == cnt++, "word id needs to be continuous starting from 2");
    word2id[p.second] = p.first;
  }

  *vocab = std::make_shared<Vocab>(std::move(word2id));
  return Status::OK();
}
const std::vector<WordType> Vocab::reserved_token_str_ = {"<pad>", "<unk>"};
}  // namespace dataset
}  // namespace mindspore
