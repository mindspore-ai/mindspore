/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/datasetops/source/udpos_op.h"

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <utility>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/wait_post.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
UDPOSOp::UDPOSOp(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
                 std::unique_ptr<DataSchema> schema, const std::vector<std::string> &udpos_files_list,
                 int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id)
    : TextFileOp(num_workers, total_rows, worker_connector_size, std::move(schema), udpos_files_list, op_connector_size,
                 shuffle_files, num_devices, device_id) {}

// A print method typically used for debugging.
void UDPOSOp::Print(std::ostream &out, bool show_all) const {
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nRow count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nUDPOS files list:\n";
    for (size_t i = 0; i < text_files_list_.size(); ++i) {
      out << " " << text_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status UDPOSOp::LoadTensor(const std::vector<std::string> &column, TensorRow *out_row, size_t index) {
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(column, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

// Function to split string based on a character delimiter.
std::vector<std::string> UDPOSOp::Split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

// Removes excess space before and after the string.
std::string UDPOSOp::Strip(const std::string &str) {
  size_t strlen = str.size();
  size_t i, j;
  i = 0;
  while (i < strlen && str[i] == ' ') {
    i++;
  }
  j = strlen - 1;
  while (j >= i && str[j] == ' ') {
    j--;
  }
  j++;
  if (i == 0 && j == strlen) {
    return str;
  } else {
    return str.substr(i, j - i);
  }
}

Status UDPOSOp::Load(const std::vector<std::string> &word, const std::vector<std::string> &universal,
                     const std::vector<std::string> &stanford, const std::string &file, int32_t worker_id) {
  size_t row_line = 3;
  size_t word_line = 0, universal_line = 1, stanford_line = 2;
  TensorRow tRow(row_line, nullptr);
  // Add file path info.
  std::vector<std::string> file_path(row_line, file);
  tRow.setPath(file_path);
  RETURN_IF_NOT_OK(LoadTensor(word, &tRow, word_line));
  RETURN_IF_NOT_OK(LoadTensor(universal, &tRow, universal_line));
  RETURN_IF_NOT_OK(LoadTensor(stanford, &tRow, stanford_line));
  RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(tRow)));
  return Status::OK();
}

Status UDPOSOp::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " + DatasetName() + " dataset dir: " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + DatasetName() + " dataset dir: " + file + " does not exist.");
  }
  std::ifstream handle(realpath.value());
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + DatasetName() + ": " + file);
  }
  int64_t rows_total = 0;
  std::string line;
  std::vector<std::string> word_column;
  std::vector<std::string> universal_column;
  std::vector<std::string> stanford_column;
  while (getline(handle, line)) {
    if (line.empty() && rows_total < start_offset) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      if (word_column.size() != 0) {
        RETURN_IF_NOT_OK(Load(word_column, universal_column, stanford_column, file, worker_id));
      }
      std::vector<std::string>().swap(word_column);
      std::vector<std::string>().swap(universal_column);
      std::vector<std::string>().swap(stanford_column);
      break;
    }
    // Skip line before start offset.
    if (rows_total < start_offset) {
      rows_total++;
      continue;
    }
    line = Strip(line);
    if (line.empty() && rows_total >= start_offset) {
      if (word_column.size() != 0) {
        RETURN_IF_NOT_OK(Load(word_column, universal_column, stanford_column, file, worker_id));
      }
      std::vector<std::string>().swap(word_column);
      std::vector<std::string>().swap(universal_column);
      std::vector<std::string>().swap(stanford_column);
      continue;
    } else if (!line.empty() && rows_total >= start_offset) {
      std::vector<std::string> column = Split(line, '\t');
      size_t right_column_size = 3;
      if (column.size() < right_column_size) {
        handle.close();
        MS_LOG(ERROR)
          << "Invalid file content, each line should contain three columns, representing word, universal and stanford.";
        RETURN_STATUS_UNEXPECTED(
          "Invalid file content, each line should contain three columns, representing word, universal and stanford.");
      }
      size_t word_line = 0, universal_line = 1, stanford_line = 2;
      word_column.push_back(column[word_line]);
      universal_column.push_back(column[universal_line]);
      stanford_column.push_back(column[stanford_line]);
    } else {
      handle.close();
      RETURN_STATUS_UNEXPECTED("[Internal ERROR], rows_total is less than start_offset.");
    }
    rows_total++;
  }
  handle.close();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
