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

#ifndef MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_SENTENCE_PIECE_VOCAB_H_
#define MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_SENTENCE_PIECE_VOCAB_H_

#include <string>
#include <memory>
#include <vector>
#include <unordered_map>
#include "minddata/dataset/util/status.h"
#include "minddata/dataset/include/constants.h"

namespace mindspore {
namespace dataset {

class SentencePieceVocab {
 public:
  static Status BuildFromFile(const std::vector<std::string> &path_list, const int32_t vocab_size,
                              const float character_coverage, const SentencePieceModel model_type,
                              const std::unordered_map<std::string, std::string> &params,
                              std::shared_ptr<SentencePieceVocab> *vocab);
  static Status SaveModel(const std::shared_ptr<SentencePieceVocab> *vocab, std::string path, std::string filename);
  SentencePieceVocab();

  ~SentencePieceVocab() = default;

  const std::string &model_proto();

  void set_model_proto(const std::string model_proto);

 private:
  std::string model_proto_;
};
}  // namespace dataset
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_MINDDATA_DATASET_TEXT_SENTENCE_PIECE_VOCAB_H_
