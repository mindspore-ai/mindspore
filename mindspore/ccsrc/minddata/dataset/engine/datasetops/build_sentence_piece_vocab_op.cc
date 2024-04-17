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

#include "minddata/dataset/engine/datasetops/build_sentence_piece_vocab_op.h"

#include "minddata/dataset/core/config_manager.h"

namespace mindspore {
namespace dataset {
BuildSentencePieceVocabOp::BuildSentencePieceVocabOp(std::shared_ptr<SentencePieceVocab> vocab,
                                                     const std::vector<std::string> col_names, int32_t vocab_size,
                                                     float character_coverage, SentencePieceModel model_type,
                                                     const std::unordered_map<std::string, std::string> &params,
                                                     int32_t op_conn_size)
    : PipelineOp(op_conn_size),
      vocab_size_(vocab_size),
      vocab_(vocab),
      col_names_(col_names),
      character_coverage_(character_coverage),
      model_type_(model_type),
      params_(params),
      col_id_(0) {
  sentence_queue_ = std::make_unique<Queue<TensorRow>>(op_conn_size);
  read_done_ = false;
  ret_status_ = Status::OK();
}

Status BuildSentencePieceVocabOp::operator()() {
  if (tree_ == nullptr) {
    RETURN_STATUS_UNEXPECTED("[Internal ERROR] Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(sentence_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(tree_->AllTasks()->CreateAsyncTask(
    "sentenceTask", std::bind(&BuildSentencePieceVocabOp::SentenceThread, this), nullptr, id()));
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));

  bool eoe_warning = false;  // give out warning if receive more than 1 eoe
  while (child_iterator_->EofHandled() == false) {
    while (new_row.empty() == false) {
      RETURN_IF_NOT_OK(sentence_queue_->EmplaceBack(new_row));
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }
    RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    CHECK_FAIL_RETURN_UNEXPECTED(!eoe_warning, "'build_sentencepiece_vocab' does not support 'repeat'.)");
    eoe_warning = true;
  }
  // add empty tensorRow for quit
  TensorRow empty_row = {};
  RETURN_IF_NOT_OK(sentence_queue_->EmplaceBack(empty_row));
  return Status::OK();
}

Status BuildSentencePieceVocabOp::SentenceThread() {
  TaskManager::FindMe()->Post();
  if (col_names_.empty() == true) {
    auto itr = column_name_id_map_.find("text");
    CHECK_FAIL_RETURN_UNEXPECTED(itr != column_name_id_map_.end(), "Invalid data, 'text' column does not exist.");
    col_id_ = itr->second;
  } else {
    auto itr = column_name_id_map_.find(col_names_[0]);
    CHECK_FAIL_RETURN_UNEXPECTED(itr != column_name_id_map_.end(), "Invalid column, column name: " + col_names_[0] +
                                                                     " does not exist, check existed "
                                                                     "column with dataset API 'get_col_names'");
    col_id_ = itr->second;
  }
  std::unique_ptr<DatasetSentenceIterator> sentence_iter = std::make_unique<DatasetSentenceIterator>(this);
  std::string model_proto;
  sentencepiece::util::Status s_status =
    sentencepiece::SentencePieceTrainer::Train(BuildParams(), sentence_iter.get(), &model_proto);
  if (!s_status.ok()) {
    RETURN_STATUS_UNEXPECTED(s_status.message());
  } else {
    if (vocab_ == nullptr) {
      RETURN_STATUS_UNEXPECTED("[Internal ERROR] SentencePiece vocab should not be null.");
    }
    vocab_->set_model_proto(model_proto);
  }
  RETURN_IF_NOT_OK(out_connector_->SendEOE());
  RETURN_IF_NOT_OK(out_connector_->SendEOF());
  return Status::OK();
}

std::unordered_map<std::string, std::string> BuildSentencePieceVocabOp::BuildParams() {
  std::unordered_map<std::string, std::string> ret_params;
  ret_params["vocab_size"] = std::to_string(vocab_size_);
  ret_params["character_coverage"] = std::to_string(character_coverage_);
  if (model_type_ == SentencePieceModel::kBpe) {
    ret_params["model_type"] = "BPE";
  } else if (model_type_ == SentencePieceModel::kChar) {
    ret_params["model_type"] = "CHAR";
  } else if (model_type_ == SentencePieceModel::kWord) {
    ret_params["model_type"] = "WORD";
  } else {
    ret_params["model_type"] = "UNIGRAM";
  }
  // filter some params that set by function param
  // filter  model_prefix that must be empty
  for (auto param : params_) {
    std::string key = param.first;
    if (key == "input" || key == "vocab_size" || key == "model_prefix" || key == "character_coverage" ||
        key == "model_type") {
      continue;
    }
    ret_params[key] = param.second;
  }

  ret_params["model_prefix"] = "";
  ret_params["minloglevel"] = "1";
  return ret_params;
}

bool BuildSentencePieceVocabOp::Done() { return read_done_; }

void BuildSentencePieceVocabOp::Next(std::string *sentence) {
  if (sentence == nullptr) {
    MS_LOG(ERROR) << "[Internal ERROR] BuildSentencePieceVocab get nullptr element, please check data.";
    return;
  }
  TensorRow new_row;
  Status s = sentence_queue_->PopFront(&new_row);

  if (s.IsError()) {
    read_done_ = true;
    ret_status_ = s;
    return;
  }
  if (new_row.empty() == true) {
    read_done_ = true;
    ret_status_ = Status::OK();
    return;
  }

  if (new_row[col_id_]->type().IsNumeric() || new_row[col_id_]->Rank() > 1) {
    ret_status_ =
      STATUS_ERROR(StatusCode::kMDUnexpectedError,
                   "Invalid data, build_sentence_piece_vocab only supports string data with rank equal to 1, "
                   "but got type: " +
                     new_row[col_id_]->type().ToString() + ", rank: " + std::to_string(new_row[col_id_]->Rank()));
    read_done_ = true;
    return;
  }

  std::string_view sentence_v;
  ret_status_ = new_row[col_id_]->GetItemAt(&sentence_v, {});
  if (ret_status_.IsError()) {
    read_done_ = true;
    return;
  }

  std::string st{sentence_v};
  *sentence = st;
  ret_status_ = Status::OK();
}

BuildSentencePieceVocabOp::DatasetSentenceIterator::DatasetSentenceIterator(BuildSentencePieceVocabOp *s_p_vocab_ptr)
    : s_p_vocab_ptr_(s_p_vocab_ptr) {}

bool BuildSentencePieceVocabOp::DatasetSentenceIterator::done() const {
  if (s_p_vocab_ptr_ == nullptr) {
    return true;
  }
  return s_p_vocab_ptr_->Done();
}

void BuildSentencePieceVocabOp::DatasetSentenceIterator::Next() {
  if (s_p_vocab_ptr_ == nullptr) {
    return;
  }
  s_p_vocab_ptr_->Next(&value_);
}
}  // namespace dataset
}  // namespace mindspore
