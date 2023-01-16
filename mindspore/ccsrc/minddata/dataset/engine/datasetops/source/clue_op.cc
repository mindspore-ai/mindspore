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
#include "minddata/dataset/engine/datasetops/source/clue_op.h"

#include <string>
#include <utility>
#include <vector>
#include <fstream>
#include <iomanip>

#include "utils/file_utils.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/util/random.h"

namespace mindspore {
namespace dataset {
ClueOp::ClueOp(int32_t num_workers, int64_t num_samples, int32_t worker_connector_size, ColKeyMap cols_to_keyword,
               std::vector<std::string> clue_files_list, int32_t op_connector_size, bool shuffle_files,
               int32_t num_devices, int32_t device_id)
    : NonMappableLeafOp(num_workers, worker_connector_size, num_samples, op_connector_size, shuffle_files, num_devices,
                        device_id),
      clue_files_list_(std::move(clue_files_list)),
      cols_to_keyword_(std::move(cols_to_keyword)) {}

Status ClueOp::Init() {
  RETURN_IF_NOT_OK(filename_index_->insert(clue_files_list_));

  int32_t safe_queue_size = static_cast<int32_t>(std::ceil(clue_files_list_.size() / num_workers_) + 1);
  io_block_queues_.Init(num_workers_, safe_queue_size);

  jagged_rows_connector_ = std::make_unique<JaggedConnector>(num_workers_, 1, worker_connector_size_);

  return Status::OK();
}

Status ClueOp::GetValue(const nlohmann::json &js, std::vector<std::string> key_chain, std::shared_ptr<Tensor> *t) {
  nlohmann::json cursor = js;
  for (int i = 0; i < key_chain.size(); i++) {
    if (cursor.find(key_chain[i]) != cursor.end()) {
      cursor = cursor[key_chain[i]];
    } else {
      RETURN_STATUS_UNEXPECTED("Invalid json file, in given JSON file, failed to find key: " + key_chain[i]);
    }
  }
  std::string final_str = key_chain.back();
  switch (cursor.type()) {
    case nlohmann::detail::value_t::string:
      RETURN_IF_NOT_OK(Tensor::CreateScalar(cursor.get<std::string>(), t));
      break;
    case nlohmann::detail::value_t::number_integer:
      RETURN_IF_NOT_OK(Tensor::CreateScalar(cursor.get<int32_t>(), t));
      break;
    case nlohmann::detail::value_t::number_unsigned:
      RETURN_IF_NOT_OK(Tensor::CreateScalar(cursor.get<uint32_t>(), t));
      break;
    case nlohmann::detail::value_t::number_float:
      RETURN_IF_NOT_OK(Tensor::CreateScalar(cursor.get<float>(), t));
      break;
    case nlohmann::detail::value_t::array:
      RETURN_IF_NOT_OK(Tensor::CreateFromVector(cursor.get<std::vector<std::string>>(), t));
      break;
    default:
      break;
  }
  return Status::OK();
}

Status ClueOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    std::string err_msg = "Invalid file path, " + file + " does not exist.";
    LOG_AND_RETURN_STATUS_SYNTAX_ERROR(err_msg);
  }

  std::ifstream handle(realpath.value());
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + file + ", the file is damaged or permission denied.");
  }

  int64_t rows_total = 0;
  std::string line;

  while (getline(handle, line)) {
    if (line.empty()) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      break;
    }
    // Skip line before start offset.
    if (rows_total < start_offset) {
      rows_total++;
      continue;
    }

    nlohmann::json js;
    try {
      js = nlohmann::json::parse(line);
    } catch (const std::exception &err) {
      // Catch any exception and convert to Status return code
      RETURN_STATUS_UNEXPECTED("Invalid json, failed to parse " + file + ", " + std::string(err.what()));
    }
    int cols_count = cols_to_keyword_.size();
    TensorRow t_row(cols_count, nullptr);
    // Add file path info
    std::vector<std::string> file_path(cols_count, file);
    t_row.setPath(file_path);
    int cout = 0;
    for (auto &p : cols_to_keyword_) {
      std::shared_ptr<Tensor> tensor;
      RETURN_IF_NOT_OK(GetValue(js, p.second, &tensor));
      t_row[cout] = std::move(tensor);
      cout++;
    }

    rows_total++;
    RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(t_row)));
  }

  return Status::OK();
}

// A print method typically used for debugging
void ClueOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nClue files list:\n";
    for (int i = 0; i < clue_files_list_.size(); ++i) {
      out << " " << clue_files_list_[i];
    }
    out << "\n\n";
  }
}

Status ClueOp::FillIOBlockQueue(const std::vector<int64_t> &i_keys) {
  int32_t queue_index = 0;
  int64_t pre_count = 0;
  int64_t start_offset = 0;
  int64_t end_offset = 0;
  bool finish = false;
  while (!finish) {
    std::vector<std::pair<std::string, int64_t>> file_index;
    if (!i_keys.empty()) {
      for (auto it = i_keys.begin(); it != i_keys.end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
        }
        file_index.emplace_back(std::pair<std::string, int64_t>((*filename_index_)[*it], *it));
      }
    } else {
      for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
        {
          if (!GetLoadIoBlockQueue()) {
            break;
          }
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

Status ClueOp::CalculateNumRowsPerShard() {
  for (auto it = filename_index_->begin(); it != filename_index_->end(); ++it) {
    int64_t count = CountTotalRows(it.value());
    filename_numrows_[it.value()] = count;
    num_rows_ += count;
  }
  if (num_rows_ == 0) {
    std::stringstream ss;
    for (int i = 0; i < clue_files_list_.size(); ++i) {
      ss << " " << clue_files_list_[i];
    }
    std::string file_list = ss.str();
    RETURN_STATUS_UNEXPECTED(
      "Invalid data, 'CLUEDataset' API can't read the data file (interface mismatch or no data found). "
      "Check file path:" +
      file_list);
  }

  num_rows_per_shard_ = static_cast<int64_t>(std::ceil(num_rows_ * 1.0 / num_devices_));
  MS_LOG(DEBUG) << "Number rows per shard is " << num_rows_per_shard_;
  return Status::OK();
}

int64_t CountTotalRowsPerFile(const std::string &file) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file, " << file << " does not exist.";
    return 0;
  }

  std::ifstream handle(realpath.value());
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open " << file << ": the file is damaged or permission denied.";
    return 0;
  }

  std::string line;
  int64_t count = 0;
  while (getline(handle, line)) {
    if (!line.empty()) {
      count++;
    }
  }

  return count;
}

int64_t ClueOp::CountTotalRows(const std::string &file) { return CountTotalRowsPerFile(file); }

Status ClueOp::CountAllFileRows(const std::vector<std::string> &files, int64_t *count) {
  RETURN_UNEXPECTED_IF_NULL(count);
  std::shared_ptr<ClueOp> op;
  *count = 0;
  for (auto file : files) {
    *count += CountTotalRowsPerFile(file);
  }
  return Status::OK();
}

Status ClueOp::ComputeColMap() {
  // Set the column name mapping (base class field)
  if (column_name_id_map_.empty()) {
    int count = 0;
    for (auto &p : cols_to_keyword_) {
      column_name_id_map_[p.first] = count;
      count++;
    }
  } else {
    MS_LOG(WARNING) << "Column name map is already set!";
  }
  return Status::OK();
}

}  // namespace dataset
}  // namespace mindspore
