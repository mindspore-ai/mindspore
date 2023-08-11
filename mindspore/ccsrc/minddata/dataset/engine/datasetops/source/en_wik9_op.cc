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

#include "minddata/dataset/engine/datasetops/source/en_wik9_op.h"

#include <fstream>
#include <utility>

#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
EnWik9Op::EnWik9Op(int32_t num_workers, int64_t total_rows, int32_t worker_connector_size,
                   std::unique_ptr<DataSchema> data_schema, const std::vector<std::string> &file_list,
                   int32_t op_connector_size, bool shuffle_files, int32_t num_devices, int32_t device_id)
    : TextFileOp(num_workers, total_rows, worker_connector_size, std::move(data_schema), file_list, op_connector_size,
                 shuffle_files, num_devices, device_id) {}

// A print method typically used for debugging.
void EnWik9Op::Print(std::ostream &out, bool show_all) const {
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
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nEnWik9 file path:\n";
    for (size_t i = 0; i < text_files_list_.size(); ++i) {
      // Print the name of per file path.
      out << " " << text_files_list_[i];
    }
    out << "\nData Schema:\n";
    out << *data_schema_ << "\n\n";
  }
}

Status EnWik9Op::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file + " does not exist.");
  }

  std::ifstream handle(realpath.value(), std::ios::in);
  if (!handle.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open file: " + file +
                             ". Check if the file is damaged or permission denied.");
  }

  int64_t rows_total = 0;
  std::string line;

  while (getline(handle, line)) {
    if (line.empty()) {
      line = "";
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

    TensorRow tRow(1, nullptr);
    tRow.setPath({file});
    auto s = LoadTensor(line, &tRow);
    if (s != Status::OK()) {
      handle.close();
      return s;
    }
    s = jagged_rows_connector_->Add(worker_id, std::move(tRow));
    if (s != Status::OK()) {
      handle.close();
      return s;
    }

    rows_total++;
  }
  handle.close();

  return Status::OK();
}

int64_t EnWik9Op::CountTotalRows(const std::string &file) {
  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file, " << file << " does not exist.";
    return 0;
  }

  std::ifstream handle(realpath.value(), std::ios::in);
  if (!handle.is_open()) {
    MS_LOG(ERROR) << "Invalid file, failed to open file: " << file
                  << ". Check if the file is damaged or permission denied.";
    return 0;
  }

  std::string line;
  int64_t count = 0;
  while (getline(handle, line)) {
    count++;
  }
  handle.close();

  return count;
}
}  // namespace dataset
}  // namespace mindspore
