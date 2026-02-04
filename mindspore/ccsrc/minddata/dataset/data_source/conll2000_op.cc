/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/data_source/conll2000_op.h"

#include <fstream>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/data_source/io_block.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/random.h"
#include "minddata/dataset/util/wait_post.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
CoNLL2000Op::CoNLL2000Op(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
                         std::unique_ptr<DataSchema> schema, const std::vector<std::string> &conll2000_file_list,
                         int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id)
    : TextFileOp(num_workers, total_rows, worker_connector_size, std::move(schema), conll2000_file_list,
                 op_connector_size, shuffle_files, num_devices, device_id) {}

// A print method typically used for debugging.
void CoNLL2000Op::Print(std::ostream &out, bool show_all) const {
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
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nCoNLL2000 file list:\n";
    for (size_t i = 0; i < text_files_list_.size(); ++i) {
      out << " " << text_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status CoNLL2000Op::LoadTensor(const std::vector<std::string> &column, TensorRow *out_row, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateFromVector(column, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

// Function to split string based on a character delimiter.
std::vector<std::string> CoNLL2000Op::Split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;

  while (getline(ss, item, delim)) {
    res.push_back(item);
  }
  return res;
}

// Removes excess space before and after the string.
std::string CoNLL2000Op::Strip(const std::string &str) {
  std::int64_t strlen = str.size();
  std::int64_t i, j;
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

Status CoNLL2000Op::Load(const std::vector<std::string> &word, const std::vector<std::string> &pos_tag,
                         const std::vector<std::string> &chunk_tag, const std::string &file, int32_t worker_id) {
  size_t row_line = 3;
  TensorRow tRow(row_line, nullptr);
  // Add file path info.
  std::vector<std::string> file_path(row_line, file);
  tRow.setPath(file_path);
  size_t word_index = 0, pos_tag_index = 1, chunk_tag_index = 2;
  RETURN_IF_NOT_OK(LoadTensor(word, &tRow, word_index));
  RETURN_IF_NOT_OK(LoadTensor(pos_tag, &tRow, pos_tag_index));
  RETURN_IF_NOT_OK(LoadTensor(chunk_tag, &tRow, chunk_tag_index));
  RETURN_IF_NOT_OK(jagged_rows_connector_->Add(worker_id, std::move(tRow)));
  return Status::OK();
}

Status CoNLL2000Op::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << DatasetName() << " dataset dir: " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + DatasetName() + " dataset dir: " + file + " does not exist.");
  }
  std::ifstream handle(realpath.value(), std::ios::in);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + DatasetName() + ": " + file);
  }
  int64_t rows_total = 0;
  std::string line;
  std::vector<std::string> word_column;
  std::vector<std::string> pos_tag_column;
  std::vector<std::string> chunk_tag_column;
  while (getline(handle, line)) {
    if (line.empty() && rows_total < start_offset) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (rows_total >= end_offset) {
      if (word_column.size() != 0) {
        Status s = Load(word_column, pos_tag_column, chunk_tag_column, file, worker_id);
        if (s.IsError()) {
          handle.close();
          return s;
        }
      }
      std::vector<std::string>().swap(word_column);
      std::vector<std::string>().swap(pos_tag_column);
      std::vector<std::string>().swap(chunk_tag_column);
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
        Status s = Load(word_column, pos_tag_column, chunk_tag_column, file, worker_id);
        if (s.IsError()) {
          handle.close();
          return s;
        }
      }
      std::vector<std::string>().swap(word_column);
      std::vector<std::string>().swap(pos_tag_column);
      std::vector<std::string>().swap(chunk_tag_column);
      continue;
    } else if (!line.empty() && rows_total >= start_offset) {
      std::vector<std::string> column = Split(line, ' ');
      size_t word_index = 0, pos_tag_index = 1, chunk_tag_index = 2;
      word_column.push_back(column[word_index]);
      pos_tag_column.push_back(column[pos_tag_index]);
      chunk_tag_column.push_back(column[chunk_tag_index]);
    }
    rows_total++;
  }
  handle.close();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
