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
#include "minddata/dataset/engine/datasetops/source/multi30k_op.h"

#include <fstream>
#include <iomanip>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "include/common/debug/common.h"
#include "minddata/dataset/engine/datasetops/source/io_block.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
// constructor of Multi30k.
Multi30kOp::Multi30kOp(int32_t num_workers, int64_t num_samples, const std::vector<std::string> &language_pair,
                       int32_t worker_connector_size, std::unique_ptr<DataSchema> schema,
                       const std::vector<std::string> &text_files_list, int32_t op_connector_size, bool shuffle_files,
                       int32_t num_devices, int32_t device_id)
    : TextFileOp(num_workers, num_samples, worker_connector_size, std::move(schema), std::move(text_files_list),
                 op_connector_size, shuffle_files, num_devices, device_id),
      language_pair_(language_pair) {}

// Print info of operator.
void Multi30kOp::Print(std::ostream &out, bool show_all) {
  // Print parameter to debug function.
  std::vector<std::string> multi30k_files_list = TextFileOp::FileNames();
  if (!show_all) {
    // Call the super class for displaying any common 1-liner info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal 1-liner info for this op.
    out << "\n";
  } else {
    // Call the super class for displaying any common detailed info.
    ParallelOp::Print(out, show_all);
    // Then show any custom derived-internal stuff.
    out << "\nSample count: " << total_rows_ << "\nDevice id: " << device_id_ << "\nNumber of devices: " << num_devices_
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nMulti30k files list:\n";
    for (int i = 0; i < multi30k_files_list.size(); ++i) {
      out << " " << multi30k_files_list[i];
    }
    out << "\n\n";
  }
}

Status Multi30kOp::LoadTensor(const std::string &line, TensorRow *out_row, size_t index) {
  RETURN_UNEXPECTED_IF_NULL(out_row);
  std::shared_ptr<Tensor> tensor;
  RETURN_IF_NOT_OK(Tensor::CreateScalar(line, &tensor));
  (*out_row)[index] = std::move(tensor);
  return Status::OK();
}

Status Multi30kOp::LoadFile(const std::string &file_en, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  auto realpath_en = FileUtils::GetRealPath(file_en.c_str());
  if (!realpath_en.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << DatasetName() + " Dataset file: " << file_en << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + DatasetName() + " Dataset file: " + file_en + " does not exist.");
  }

  // We use English files to find Germany files, to make sure that data are ordered.
  Path path_en(file_en);
  Path parent_path(path_en.ParentPath());
  std::string basename = path_en.Basename();
  int suffix_len = 3;
  std::string suffix_de = ".de";
  auto pos = basename.find(".");
  CHECK_FAIL_RETURN_UNEXPECTED(pos != std::string::npos, "Invalid file, can not parse dataset file:" + file_en);
  basename = basename.replace(pos, suffix_len, suffix_de);
  Path BaseName(basename);
  Path path_de = parent_path / BaseName;
  std::string file_de = path_de.ToString();
  auto realpath_de = FileUtils::GetRealPath(file_de.c_str());
  if (!realpath_de.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << DatasetName() + " Dataset file: " << file_de << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + DatasetName() + " Dataset file: " + file_de + " does not exist.");
  }

  std::ifstream handle_en(realpath_en.value());
  CHECK_FAIL_RETURN_UNEXPECTED(handle_en.is_open(), "Invalid file, failed to open en file: " + file_en);
  std::ifstream handle_de(realpath_de.value());
  CHECK_FAIL_RETURN_UNEXPECTED(handle_de.is_open(), "Invalid file, failed to open de file: " + file_de);

  // Set path for path in class TensorRow.
  std::string line_en;
  std::string line_de;
  std::vector<std::string> path = {file_en, file_de};

  int row_total = 0;
  while (getline(handle_en, line_en) && getline(handle_de, line_de)) {
    if (line_en.empty() && line_de.empty()) {
      continue;
    }
    // If read to the end offset of this file, break.
    if (row_total >= end_offset) {
      break;
    }
    // Skip line before start offset.
    if (row_total < start_offset) {
      ++row_total;
      continue;
    }

    int tensor_size = 2;
    TensorRow tRow(tensor_size, nullptr);

    Status rc_en;
    Status rc_de;
    if (language_pair_[0] == "en") {
      rc_en = LoadTensor(line_en, &tRow, 0);
      rc_de = LoadTensor(line_de, &tRow, 1);
    } else if (language_pair_[0] == "de") {
      rc_en = LoadTensor(line_en, &tRow, 1);
      rc_de = LoadTensor(line_de, &tRow, 0);
    }
    if (rc_en.IsError() || rc_de.IsError()) {
      handle_en.close();
      handle_de.close();
      RETURN_IF_NOT_OK(rc_en);
      RETURN_IF_NOT_OK(rc_de);
    }
    (&tRow)->setPath(path);

    Status rc = jagged_rows_connector_->Add(worker_id, std::move(tRow));
    if (rc.IsError()) {
      handle_en.close();
      handle_de.close();
      return rc;
    }
    ++row_total;
  }

  handle_en.close();
  handle_de.close();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
