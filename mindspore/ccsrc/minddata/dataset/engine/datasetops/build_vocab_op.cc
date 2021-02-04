/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/build_vocab_op.h"

#include <algorithm>
#include <iomanip>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include "minddata/dataset/core/config_manager.h"

namespace mindspore {
namespace dataset {

BuildVocabOp::BuildVocabOp(std::shared_ptr<Vocab> vocab, std::vector<std::string> col_names,
                           std::pair<int64_t, int64_t> freq_r, int64_t top_k, const std::vector<std::string> &tokens,
                           bool prepend, int32_t num_workers, int32_t op_conn_size)
    : ParallelOp(num_workers, op_conn_size),
      interval_(op_conn_size * num_workers),
      vocab_(vocab),
      col_names_(col_names),
      freq_range_(freq_r),
      top_k_(top_k),
      special_tokens_(tokens),
      special_first_(prepend) {
  // init two queues for thread sync
  distributor_queue_ = std::make_unique<Queue<TensorRow>>(num_workers * op_conn_size);
  collector_queue_ =
    std::make_unique<Queue<std::unique_ptr<std::unordered_map<std::string, int64_t>>>>(num_workers * op_conn_size);
}

Status BuildVocabOp::WorkerEntry(int32_t worker_id) {
  TaskManager::FindMe()->Post();
  TensorRow new_row;
  RETURN_IF_NOT_OK(distributor_queue_->PopFront(&new_row));
  std::unique_ptr<std::unordered_map<std::string, int64_t>> wrkr_map =
    std::make_unique<std::unordered_map<std::string, int64_t>>();
  int32_t row_cnt = 0;
  while (!new_row.empty()) {
    for (int32_t col : col_ids_) {
      CHECK_FAIL_RETURN_UNEXPECTED(!new_row[col]->type().IsNumeric(),
                                   "Invalid data, build_vocab only works on string data, but got numeric data type: " +
                                     new_row[col]->type().ToString());
      for (auto itr = new_row[col]->begin<std::string_view>(); itr != new_row[col]->end<std::string_view>(); itr++) {
        (*wrkr_map)[std::string(*itr)] += 1;
      }
    }
    row_cnt++;  // row is processed by this point
    if ((row_cnt % interval_ == 0) && ((row_cnt / interval_) % num_workers_ == worker_id) && (!wrkr_map->empty())) {
      RETURN_IF_NOT_OK(collector_queue_->Add(std::move(wrkr_map)));
      wrkr_map = std::make_unique<std::unordered_map<std::string, int64_t>>();
    }
    RETURN_IF_NOT_OK(distributor_queue_->PopFront(&new_row));
  }
  // clean up
  if (!wrkr_map->empty()) {
    RETURN_IF_NOT_OK(collector_queue_->Add(std::move(wrkr_map)));
  }
  // empty map as quit signal
  RETURN_IF_NOT_OK(collector_queue_->Add(std::make_unique<std::unordered_map<std::string, int64_t>>()));
  return Status::OK();
}

Status BuildVocabOp::operator()() {
  // launch the collector thread
  if (tree_ == nullptr) {
    return Status(StatusCode::kMDUnexpectedError, __LINE__, __FILE__, "Pipeline init failed, Execution tree not set.");
  }
  RETURN_IF_NOT_OK(distributor_queue_->Register(tree_->AllTasks()));
  RETURN_IF_NOT_OK(collector_queue_->Register(tree_->AllTasks()));
  // launch worker threads and collector thread
  RETURN_IF_NOT_OK(
    tree_->LaunchWorkers(num_workers_, std::bind(&BuildVocabOp::WorkerEntry, this, std::placeholders::_1), "", id()));
  RETURN_IF_NOT_OK(
    tree_->AllTasks()->CreateAsyncTask("collector", std::bind(&BuildVocabOp::CollectorThread, this), nullptr, id()));
  TaskManager::FindMe()->Post();
  child_iterator_ = std::make_unique<ChildIterator>(this, 0, 0);
  TensorRow new_row;
  RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
  if (!col_names_.empty()) {
    col_ids_.reserve(col_names_.size());
    for (std::string col : col_names_) {
      auto itr = column_name_id_map_.find(col);
      CHECK_FAIL_RETURN_UNEXPECTED(itr != column_name_id_map_.end(),
                                   "Invalid parameter, column name: " + col + " does not exist.");
      col_ids_.push_back(itr->second);
    }
  } else {
    col_ids_.reserve(column_name_id_map_.size());
    for (const auto &p : column_name_id_map_) {
      col_ids_.push_back(p.second);
    }
  }
  bool eoe_warning = false;  // give out warning if receive more than 1 eoe
  while (child_iterator_->eof_handled() == false) {
    while (new_row.empty() == false) {
      RETURN_IF_NOT_OK(distributor_queue_->EmplaceBack(new_row));
      RETURN_IF_NOT_OK(child_iterator_->FetchNextTensorRow(&new_row));
    }
    CHECK_FAIL_RETURN_UNEXPECTED(!eoe_warning, "no op should be after from_dataset (repeat detected)");
    eoe_warning = true;
  }

  // tell all workers to quit
  for (int32_t wrkr_id = 0; wrkr_id < num_workers_; wrkr_id++) {
    RETURN_IF_NOT_OK(distributor_queue_->EmplaceBack(TensorRow()));
  }
  return Status::OK();
}

Status BuildVocabOp::CollectorThread() {
  TaskManager::FindMe()->Post();
  int32_t num_quited_worker = 0;
  std::unique_ptr<std::unordered_map<std::string, int64_t>> wrkr_map;
  while (num_quited_worker != num_workers_) {
    RETURN_IF_NOT_OK(collector_queue_->PopFront(&wrkr_map));
    RETURN_UNEXPECTED_IF_NULL(wrkr_map);
    if (!wrkr_map->empty()) {
      for (const auto &wd : *wrkr_map) word_cnt_[wd.first] += wd.second;
    } else {
      ++num_quited_worker;
    }
  }  // all frequencies are obtained
  CHECK_FAIL_RETURN_UNEXPECTED(!word_cnt_.empty(), "Invalid data, no words in the dataset.");
  std::vector<std::string> words;
  // make sure enough is reserved, this will become a partially sorted list eventually
  words.reserve(wrkr_map->size());

  for (auto it = word_cnt_.begin(); it != word_cnt_.end();) {
    if (it->second >= freq_range_.first && it->second <= freq_range_.second) {
      words.push_back(it->first);
      it++;
    } else {
      it = word_cnt_.erase(it);
    }
  }
  std::string err_msg;

  for (const std::string &sp_tk : special_tokens_) {
    // if a special word exists in dataset, warn user about this
    err_msg += (word_cnt_.find(sp_tk) != word_cnt_.end() ? sp_tk + "\t" : "");
  }

  CHECK_FAIL_RETURN_UNEXPECTED(err_msg.empty(),
                               "Invalid data, these special words are already in the dataset: " + err_msg + ".");

  int64_t num_words = std::min(static_cast<int64_t>(words.size()), top_k_);
  if (num_words == 0) {
    MS_LOG(WARNING) << "No word falls in the frequency range: (" << freq_range_.first << "," << freq_range_.second
                    << ") vocab would be empty (except for special tokens).";
  }

  // this would take the top-k most frequent words
  std::partial_sort(words.begin(), words.begin() + num_words, words.end(),
                    [this](const std::string &w1, const std::string &w2) {
                      int64_t f1 = word_cnt_[w1], f2 = word_cnt_[w2];
                      return f1 == f2 ? w1 < w2 : f1 > f2;
                    });

  if (special_first_) {
    for (const std::string &sp_tk : special_tokens_) vocab_->append_word(sp_tk);
  }

  for (int64_t i = 0; i < num_words; i++) {
    vocab_->append_word(words[i]);
  }

  if (!special_first_) {
    for (const std::string &sp_tk : special_tokens_) vocab_->append_word(sp_tk);
  }

  RETURN_IF_NOT_OK(out_connector_->Add(0, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOE)));
  RETURN_IF_NOT_OK(out_connector_->Add(0, std::make_unique<DataBuffer>(0, DataBuffer::kDeBFlagEOF)));
  // then use std::nth_element to partial sort
  return Status::OK();
}

Status BuildVocabOp::Builder::Build(std::shared_ptr<BuildVocabOp> *op) {
  CHECK_FAIL_RETURN_UNEXPECTED(
    builder_num_workers_ > 0,
    "Invalid parameter, num_parallel_workers must be greater than 0, but got " + std::to_string(builder_num_workers_));
  CHECK_FAIL_RETURN_UNEXPECTED(
    builder_top_k_ > 0, "Invalid parameter, top_k must be greater than 0, but got " + std::to_string(builder_top_k_));
  CHECK_FAIL_RETURN_UNEXPECTED(builder_max_freq_ >= builder_min_freq_ && builder_min_freq_ >= 0,
                               "Invalid parameter, frequency range [a,b] must be 0 <= a <= b (a,b are inclusive).");
  (*op) = std::make_shared<BuildVocabOp>(
    builder_vocab_, builder_col_names_, std::make_pair(builder_min_freq_, builder_max_freq_), builder_top_k_,
    builder_speical_tokens_, builder_special_first_, builder_num_workers_, builder_connector_size_);
  return Status::OK();
}

BuildVocabOp::Builder::Builder()
    : builder_top_k_(std::numeric_limits<int64_t>::max()),
      builder_min_freq_(0),
      builder_max_freq_(std::numeric_limits<int64_t>::max()),
      builder_special_first_(true) {
  std::shared_ptr<ConfigManager> cfg = GlobalContext::config_manager();
  builder_num_workers_ = cfg->num_parallel_workers();
  builder_connector_size_ = cfg->op_connector_size();
}

// A print method typically used for debugging
void BuildVocabOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nCode is needed here to show more info about the op."
        << "\n\n";
  }
}
}  // namespace dataset
}  // namespace mindspore
