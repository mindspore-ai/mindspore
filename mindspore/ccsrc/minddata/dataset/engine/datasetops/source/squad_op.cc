/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/squad_op.h"

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
SQuADOp::SQuADOp(const std::string &dataset_dir, const std::string &usage, int32_t num_workers, int64_t num_samples,
                 int32_t worker_connector_size, std::unique_ptr<DataSchema> schema, int32_t op_connector_size,
                 bool shuffle_files, int32_t num_devices, int32_t device_id)
    : NonMappableLeafOp(num_workers, worker_connector_size, num_samples, op_connector_size, shuffle_files, num_devices,
                        device_id),
      dataset_dir_(std::move(dataset_dir)),
      usage_(std::move(usage)),
      data_schema_(std::move(schema)) {}

Status SQuADOp::Init() {
  RETURN_IF_NOT_OK(GetFiles(dataset_dir_, usage_, &squad_files_list_));
  RETURN_IF_NOT_OK(filename_index_->insert(squad_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(squad_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);

  return Status::OK();
}

void SQuADOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Show any custom derived-internal stuff.
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nSQuAD files list:\n";
    for (int i = 0; i < squad_files_list_.size(); ++i) {
      out << " " << squad_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status SQuADOp::GetFiles(const std::string &dataset_dir, const std::string &usage,
                         std::vector<std::string> *data_files_list) {
  RETURN_UNEXPECTED_IF_NULL(data_files_list);
  auto real_dataset_dir = FileUtils::GetRealPath(dataset_dir.c_str());
  if (!real_dataset_dir.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, SQuAD Dataset dir: " << dataset_dir << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, SQuAD Dataset dir: " + dataset_dir + " does not exist.");
  }
  Path root_dir(real_dataset_dir.value());

  const Path train_file_v1("train-v1.1.json");
  const Path train_file_v2("train-v2.0.json");
  const Path dev_file_v1("dev-v1.1.json");
  const Path dev_file_v2("dev-v2.0.json");

  if (usage == "train" || usage == "all") {
    Path train_path_v1 = root_dir / train_file_v1;
    Path train_path_v2 = root_dir / train_file_v2;
    bool train_check_v1 = train_path_v1.Exists() && !train_path_v1.IsDirectory();
    bool train_check_v2 = train_path_v2.Exists() && !train_path_v2.IsDirectory();
    CHECK_FAIL_RETURN_UNEXPECTED(train_check_v1 || train_check_v2,
                                 "Invalid path, failed to find SQuAD train data file: " + dataset_dir);
    if (train_check_v1) {
      data_files_list->emplace_back(train_path_v1.ToString());
      MS_LOG(INFO) << "SQuAD operator found train data file " << train_path_v1.ToString() << ".";
    }
    if (train_check_v2) {
      data_files_list->emplace_back(train_path_v2.ToString());
      MS_LOG(INFO) << "SQuAD operator found train data file " << train_path_v2.ToString() << ".";
    }
  }
  if (usage == "dev" || usage == "all") {
    Path dev_path_v1 = root_dir / dev_file_v1;
    Path dev_path_v2 = root_dir / dev_file_v2;
    bool dev_check_v1 = dev_path_v1.Exists() && !dev_path_v1.IsDirectory();
    bool dev_check_v2 = dev_path_v2.Exists() && !dev_path_v2.IsDirectory();
    CHECK_FAIL_RETURN_UNEXPECTED(dev_check_v1 || dev_check_v2,
                                 "Invalid path, failed to find SQuAD dev data file: " + dataset_dir);
    if (dev_check_v1) {
      data_files_list->emplace_back(dev_path_v1.ToString());
      MS_LOG(INFO) << "SQuAD operator found dev data file " << dev_path_v1.ToString() << ".";
    }
    if (dev_check_v2) {
      data_files_list->emplace_back(dev_path_v2.ToString());
      MS_LOG(INFO) << "SQuAD operator found dev data file " << dev_path_v2.ToString() << ".";
    }
  }
  return Status::OK();
}

template <typename T>
Status SQuADOp::SearchNodeInJson(const nlohmann::json &input_tree, std::string node_name, T *output_node) {
  RETURN_UNEXPECTED_IF_NULL(output_node);
  auto node = input_tree.find(node_name);
  CHECK_FAIL_RETURN_UNEXPECTED(node != input_tree.end(), "Invalid data, required node not found in JSON: " + node_name);
  (*output_node) = *node;
  return Status::OK();
}

Status SQuADOp::AnswersLoad(const nlohmann::json &answers_tree, std::vector<std::string> *answer_text_vec,
                            std::vector<uint32_t> *answer_start_vec) {
  RETURN_UNEXPECTED_IF_NULL(answer_text_vec);
  RETURN_UNEXPECTED_IF_NULL(answer_start_vec);
  for (nlohmann::json answers : answers_tree) {
    uint32_t answer_start;
    std::string answer_text;
    RETURN_IF_NOT_OK(SearchNodeInJson(answers, "text", &answer_text));
    RETURN_IF_NOT_OK(SearchNodeInJson(answers, "answer_start", &answer_start));
    answer_text_vec->push_back(answer_text);
    answer_start_vec->push_back(answer_start);
  }
  if (answer_start_vec->size() == 0) {
    answer_start_vec->push_back(-1);
  }
  if (answer_text_vec->size() == 0) {
    answer_text_vec->push_back("");
  }
  return Status::OK();
}

Status SQuADOp::LoadTensorFromScalar(const std::string &scalar_item, TensorRow *out_row, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(scalar_item, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

template <typename T>
Status SQuADOp::LoadTensorFromVector(const std::vector<T> &vector_item, TensorRow *out_row, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(vector_item, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

Status SQuADOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  std::ifstream handle(file);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file);
  }

  nlohmann::json root;
  int64_t rows_total = 0;
  int num_columns = data_schema_->NumColumns();

  try {
    handle >> root;
  } catch (const std::exception &err) {
    handle.close();
    // Catch any exception and convert to Status return code.
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse JSON file: " + file);
  }
  handle.close();

  nlohmann::json data_list;
  RETURN_IF_NOT_OK(SearchNodeInJson(root, "data", &data_list));
  for (nlohmann::json data : data_list) {
    nlohmann::json paragraphs_list;
    RETURN_IF_NOT_OK(SearchNodeInJson(data, "paragraphs", &paragraphs_list));

    for (nlohmann::json paragraphs : paragraphs_list) {
      std::string context;
      RETURN_IF_NOT_OK(SearchNodeInJson(paragraphs, "context", &context));
      nlohmann::json qas_list;
      RETURN_IF_NOT_OK(SearchNodeInJson(paragraphs, "qas", &qas_list));

      for (nlohmann::json qas : qas_list) {
        // If read to the end offset of this file, break.
        if (rows_total >= end_offset) {
          break;
        }
        // Skip line before start offset.
        if (rows_total < start_offset) {
          rows_total++;
          continue;
        }

        std::string question;
        RETURN_IF_NOT_OK(SearchNodeInJson(qas, "question", &question));
        nlohmann::json answers_list;
        RETURN_IF_NOT_OK(SearchNodeInJson(qas, "answers", &answers_list));

        std::vector<std::string> text;
        std::vector<uint32_t> answer_start;

        RETURN_IF_NOT_OK(AnswersLoad(answers_list, &text, &answer_start));

        TensorRow tRow(num_columns, nullptr);
        // Add file path info.
        std::vector<std::string> file_path(num_columns, file);
        tRow.setPath(file_path);

        int64_t context_index = 0;
        int64_t question_index = 1;
        int64_t text_index = 2;
        int64_t answer_start_index = 3;
        // Put the data into a tensor table.
        RETURN_IF_NOT_OK(LoadTensorFromScalar(context, &tRow, context_index));
        RETURN_IF_NOT_OK(LoadTensorFromScalar(question, &tRow, question_index));
        RETURN_IF_NOT_OK(LoadTensorFromVector(text, &tRow, text_index));
        RETURN_IF_NOT_OK(LoadTensorFromVector(answer_start, &tRow, answer_start_index));

        rows_total++;
        RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(tRow)));
      }
    }
  }

  return Status::OK();
}

Status SQuADOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        if (!GetLoadIoBlockQueue()) {
          break;
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        if (!GetLoadIoBlockQueue()) {
          break;
        }
        file_index.emplace_back(std::pair<std::string, int64_t>(it.value(), it.key()));
      }
    }
    for (auto file_info : file_index) {
      if (NeedPushFileToBlockQueue(file_info.first, &start_offset, &end_offset, pre_count)) {
        auto ioBlock =
          std::make_unique<FilenameBlock>(file_info.second, start_offset, end_offset, IOBlock::kDeIoBlockNone);
        RETURN_IF_NOT_OK(PushIoBlockQueue(queue_index, std::move(ioBlock)));
        queue_index = (queue_index + 1) % num_workers_;
      }

      pre_count += filename_numrows_[file_info.first];
    }

    if (pre_count < (static_cast<int64_t>(device_id_) + 1) * num_rows_per_shard_) {
      finish = false;
    } else {
      finish = true;
    }
  }

  RETURN_IF_NOT_OK(PostEndOfEpoch(queue_index));
  return Status::OK();
}

Status SQuADOp::CountTensorRowsPreFile(const std::string &file, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  std::ifstream handle(file);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file);
  }

  nlohmann::json root;
  *count = 0;

  try {
    handle >> root;
  } catch (const std::exception &err) {
    handle.close();
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse JSON file: " + file);
  }
  handle.close();

  nlohmann::json data_list;
  RETURN_IF_NOT_OK(SearchNodeInJson(root, "data", &data_list));

  for (nlohmann::json data : data_list) {
    nlohmann::json paragraphs_list;
    RETURN_IF_NOT_OK(SearchNodeInJson(data, "paragraphs", &paragraphs_list));

    for (nlohmann::json paragraphs : paragraphs_list) {
      nlohmann::json qas_list;
      RETURN_IF_NOT_OK(SearchNodeInJson(paragraphs, "qas", &qas_list));
      *count += qas_list.size();
    }
  }

  return Status::OK();
}

Status SQuADOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count;
    RETURN_IF_NOT_OK(CountTensorRowsPreFile(it.value(), &count));
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (int i = 0; i < squad_files_list_.size(); ++i) {
      ss << " " << squad_files_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, data file may not be suitable to read with SQuADDataset API. Check file path:" + file_list);
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

Status SQuADOp::CountAllFileRows(const std::string &dataset_dir, const std::string &usage, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  std::vector<std::string> files_list;
  RETURN_IF_NOT_OK(GetFiles(dataset_dir, usage, &files_list));
  *count = 0;
  for (auto file : files_list) {
    int64_t count_pre_file;
    RETURN_IF_NOT_OK(CountTensorRowsPreFile(file, &count_pre_file));
    *count += count_pre_file;
  }
  return Status::OK();
}

Status SQuADOp::ComputeColMap() {
  // Set the column name mapping (base class field).
  if (column_name_id_map_.empty()) {
    for (int32_t i = 0; i < data_schema_->NumColumns(); ++i) {
      column_name_id_map_[data_schema_->Column(i).Name()] = i;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
