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

#include "minddata/dataset/text/sentence_piece_vocab.h"

#include <sentencepiece_trainer.h>
#include <sentencepiece_processor.h>
#include <fstream>

#include "utils/ms_utils.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {

SentencePieceVocab::SentencePieceVocab() : model_proto_("") {}

Status SentencePieceVocab::BuildFromFile(const std::vector<std::string> &path_list, const int32_t vocab_size,
                                         const float character_coverage, const SentencePieceModel model_type,
                                         const std::unordered_map<std::string, std::string> &params,
                                         std::shared_ptr<SentencePieceVocab> *vocab) {
  std::unordered_map<std::string, std::string> unorder_map;

  // the input of sentence is comma separated string
  std::string input_str = "";
  for (auto path : path_list) {
    input_str += path;
    input_str += ",";
  }
  input_str.pop_back();
  unorder_map["input"] = input_str;
  unorder_map["vocab_size"] = std::to_string(vocab_size);
  unorder_map["model_prefix"] = "";
  unorder_map["minloglevel"] = "1";
  unorder_map["character_coverage"] = std::to_string(character_coverage);
  if (model_type == SentencePieceModel::kWord) {
    unorder_map["model_type"] = "WORD";
  } else if (model_type == SentencePieceModel::kBpe) {
    unorder_map["model_type"] = "BPE";
  } else if (model_type == SentencePieceModel::kChar) {
    unorder_map["model_type"] = "CHAR";
  } else {
    unorder_map["model_type"] = "UNIGRAM";
  }

  // filter some params that set by function param
  // filter  model_prefix that must be empty
  for (auto param : params) {
    std::string key = param.first;
    if (key == "input" || key == "vocab_size" || key == "model_prefix" || key == "character_coverage" ||
        key == "model_type") {
      continue;
    }
    unorder_map[key] = param.second;
  }

  // set sentence lib's log
  unorder_map["minloglevel"] = "1";
  *vocab = std::make_shared<SentencePieceVocab>();
  std::string model_proto;
  sentencepiece::util::Status s_status = sentencepiece::SentencePieceTrainer::Train(unorder_map, nullptr, &model_proto);
  if (!s_status.ok()) {
    std::string err_msg = "SentencePieceVocab: " + std::string(s_status.message());
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, err_msg);
  }
  vocab->get()->set_model_proto(model_proto);

  return Status::OK();
}

Status SentencePieceVocab::SaveModel(const std::shared_ptr<SentencePieceVocab> *vocab, std::string path,
                                     std::string filename) {
  char real_path[PATH_MAX] = {0};

  if (path.size() >= PATH_MAX) {
    RETURN_STATUS_UNEXPECTED(
      "SentencePieceVocab: sentence model path is invalid for "
      "path length longer than 4096.");
  }
#if defined(_WIN32) || defined(_WIN64)
  if (_fullpath(real_path, common::SafeCStr(path), PATH_MAX) == nullptr) {
    RETURN_STATUS_UNEXPECTED(
      "SentencePieceVocab: sentence model path is invalid for "
      "path length longer than 4096.");
  }
#else
  if (realpath(common::SafeCStr(path), real_path) == nullptr) {
    RETURN_STATUS_UNEXPECTED("SentencePieceVocab: sentence model path: " + path +
                             " is not existed or permission denied.");
  }
#endif

  std::string abs_real_path = (Path(real_path) / Path(filename)).toString();
  std::ofstream os_file(abs_real_path, std::ios::out);
  (void)os_file.write(vocab->get()->model_proto().data(), vocab->get()->model_proto().size());
  os_file.close();
  return Status::OK();
}

const std::string &SentencePieceVocab::model_proto() { return model_proto_; }

void SentencePieceVocab::set_model_proto(const std::string model_proto) { model_proto_ = model_proto; }

}  // namespace dataset
}  // namespace mindspore
