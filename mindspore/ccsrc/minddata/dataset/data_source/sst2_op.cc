/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "minddata/dataset/data_source/sst2_op.h"

#include <fstream>

#include "include/utils/common.h"
#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/engine/jagged_connector.h"
#include "minddata/dataset/util/random.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace dataset {
SST2Op::SST2Op(const std::vector<std::string> &dataset_files_list, const std::string &usage, char field_delim,
               const std::vector<std::shared_ptr<BaseRecord>> &column_default,
               const std::vector<std::string> &column_name, int32_t num_workers, int64_t num_samples,
               int32_t worker_connector_size, int32_t op_connector_size, bool shuffle_files, int32_t num_devices,
               int32_t device_id)
    : CsvOp(dataset_files_list, field_delim, column_default, column_name, num_workers, num_samples,
            worker_connector_size, op_connector_size, shuffle_files, num_devices, device_id),
      usage_(usage) {}

void SST2Op::Print(std::ostream &out, bool show_all) const {
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
        << "\nShuffle files: " << ((shuffle_files_) ? "yes" : "no") << "\nST2 files list:\n";
    for (int i = 0; i < csv_files_list_.size(); ++i) {
      out << " " << csv_files_list_[i];
    }
    out << "\n\n";
  }
}

std::vector<std::string> SST2Op::split(const std::string &s, char delim) {
  std::vector<std::string> res;
  std::stringstream ss(s);
  std::string item;
  bool skip = usage_ == "test";
  while (getline(ss, item, delim)) {
    if (skip) {
      skip = false;
    } else {
      res.push_back(item);
    }
  }
  return res;
}

Status SST2Op::LoadFile(const std::string &file, int64_t start_offset, int64_t end_offset, int32_t worker_id) {
  CsvParser csv_parser(worker_id, jagged_rows_connector_.get(), field_delim_, column_default_list_, file);
  RETURN_IF_NOT_OK(csv_parser.InitCsvParser());
  csv_parser.SetStartOffset(start_offset);
  csv_parser.SetEndOffset(end_offset);

  auto realpath = FileUtils::GetRealPath(file.c_str());
  if (!realpath.has_value()) {
    MS_LOG(ERROR) << "Invalid file path, " << file << " does not exist.";
    RETURN_STATUS_UNEXPECTED("Invalid file path, " + file + " does not exist.");
  }

  std::ifstream ifs;
  ifs.open(realpath.value(), std::ifstream::in);
  if (!ifs.is_open()) {
    RETURN_STATUS_UNEXPECTED("Invalid file, failed to open " + file + ", the file is damaged or permission denied.");
  }
  if (column_name_list_.empty()) {
    std::string tmp;
    getline(ifs, tmp);
  }
  bool skip = usage_ == "test";
  csv_parser.Reset();
  try {
    while (ifs.good()) {
      // when ifstream reaches the end of file, the function get() return std::char_traits<char>::eof()
      // which is a 32-bit -1, it's not equal to the 8-bit -1 on Euler OS. So instead of char, we use
      // int to receive its return value.
      int chr = ifs.get();
      if (skip) {
        if (chr == field_delim_) {
          skip = false;
        }
        continue;
      }
      if (usage_ == "test" && chr == '\n') {
        skip = true;
      }
      int err = csv_parser.ProcessMessage(chr);
      if (err != 0) {
        // if error code is -2, the returned error is interrupted
        if (err == -2) {
          ifs.close();
          return Status(kMDInterrupted);
        }
        ifs.close();
        RETURN_STATUS_UNEXPECTED("Invalid file, failed to parse csv file: " + file + " at line " +
                                 std::to_string(csv_parser.GetTotalRows() + 1) +
                                 ". Error message: " + csv_parser.GetErrorMessage());
      }
    }
  } catch (std::invalid_argument &ia) {
    ifs.close();
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid csv, csv file: " + file + " parse failed at line " + err_row +
                             ", type does not match.");
  } catch (std::out_of_range &oor) {
    ifs.close();
    std::string err_row = std::to_string(csv_parser.GetTotalRows() + 1);
    RETURN_STATUS_UNEXPECTED("Invalid csv, " + file + " parse failed at line " + err_row + " : value out of range.");
  }
  ifs.close();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
