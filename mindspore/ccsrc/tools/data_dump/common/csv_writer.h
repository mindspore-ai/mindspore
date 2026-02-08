/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_TOOLS_DATA_DUMP_COMMON_CSV_WRITER_H_
#define MINDSPORE_CCSRC_TOOLS_DATA_DUMP_COMMON_CSV_WRITER_H_

#include <fstream>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

#include "utils/ms_utils.h"

namespace mindspore {
class CsvFileMutexManager {
 public:
  static CsvFileMutexManager &GetInstance();
  std::mutex &GetCsvMutex(const std::string &file_name);

 private:
  std::unordered_map<std::string, std::mutex> csv_mutex_map;
  std::mutex manager_mutex;
};

class CsvWriter {
 public:
  static CsvWriter &GetInstance();

  CsvWriter() = default;
  ~CsvWriter();
  DISABLE_COPY_AND_ASSIGN(CsvWriter)
  bool OpenFile(const std::string &path, const std::string &header = "", bool trunc = false);
  void CloseFile() noexcept;
  template <typename T>
  void WriteToCsv(const T &val, bool end_line = false) {
    std::shared_lock<std::shared_mutex> lock(write_mutex_);
    file_ << val;
    if (end_line) {
      file_ << kEndLine;
      (void)file_.flush();
    } else {
      file_ << kSeparator;
    }
  }

 private:
  const std::string kSeparator = ",";
  const std::string kEndLine = "\n";
  std::ofstream file_;
  std::string file_path_str_ = "";
  std::shared_mutex write_mutex_;
};
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_TOOLS_DATA_DUMP_COMMON_CSV_WRITER_H_
