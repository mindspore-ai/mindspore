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

#include <sentencepiece_processor.h>
#include <sentencepiece_trainer.h>

#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "minddata/dataset/include/dataset/constants.h"
#include "minddata/dataset/include/dataset/text.h"
#include "minddata/dataset/util/status.h"
#include "include/api/status.h"
#ifndef BUILD_LITE
#include "mindspore/core/utils/file_utils.h"
#else
#include "mindspore/lite/src/common/file_utils.h"
#endif
#include "utils/ms_utils.h"

namespace mindspore {
namespace dataset {

SentencePieceVocab::SentencePieceVocab() : model_proto_("") {}

Status SentencePieceVocab::BuildFromFile(const std::vector<std::string> &path_list, const int32_t vocab_size,
                                         const float character_coverage, const SentencePieceModel model_type,
                                         const std::unordered_map<std::string, std::string> &params,
                                         std::shared_ptr<SentencePieceVocab> *vocab) {
  if (vocab == nullptr) {
    RETURN_STATUS_UNEXPECTED("SentencePieceVocab::BuildFromFile: input vocab can not be null");
  }
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
    RETURN_STATUS_UNEXPECTED(err_msg);
  }
  vocab->get()->set_model_proto(model_proto);

  return Status::OK();
}

Status SentencePieceVocab::SaveModel(const std::shared_ptr<SentencePieceVocab> *vocab, const std::string path,
                                     std::string filename) {
  if (vocab == nullptr) {
    RETURN_STATUS_UNEXPECTED("SentencePieceVocab::SaveModel: input vocab can not be null");
  }
  auto realpath = FileUtils::GetRealPath(path.c_str());
  if (!realpath.has_value()) {
    RETURN_STATUS_UNEXPECTED("Get real path failed, path=" + path);
  }

  std::optional<std::string> whole_path = "";
  std::optional<std::string> local_file_name = filename;
  FileUtils::ConcatDirAndFileName(&realpath, &local_file_name, &whole_path);

  std::ofstream os_file(whole_path.value(), std::ios::out);
  (void)os_file.write(vocab->get()->model_proto().data(), vocab->get()->model_proto().size());
  os_file.close();

  ChangeFileMode(whole_path.value(), S_IRUSR | S_IWUSR);

  return Status::OK();
}

const std::string &SentencePieceVocab::model_proto() { return model_proto_; }

void SentencePieceVocab::set_model_proto(const std::string model_proto) { model_proto_ = model_proto; }

}  // namespace dataset
}  // namespace mindspore
