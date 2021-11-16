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

#ifndef MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_TENSOR_STAT_DUMP_H_
#define MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_TENSOR_STAT_DUMP_H_

#include <string>
#include <fstream>

#include "utils/ms_utils.h"

namespace mindspore {
class Debugger;
class CsvWriter {
 public:
  static CsvWriter &GetInstance() {
    static CsvWriter instance;
    return instance;
  }

 private:
  const std::string kSeparator = ",";
  const std::string kEndLine = "\n";
  std::ofstream file_;
  std::string file_path_str_ = "";

 public:
  CsvWriter() = default;
  ~CsvWriter();
  DISABLE_COPY_AND_ASSIGN(CsvWriter)
  bool OpenFile(const std::string &path, const std::string &header = "");
  void CloseFile();
  template <typename T>
  void WriteToCsv(const T &val, bool end_line = false);
};

class TensorStatDump {
  static const char CSV_HEADER[];
  static const char CSV_FILE_NAME[];

  const std::string &original_kernel_name_;
  const std::string &op_type_;
  const std::string &op_name_;
  uint32_t task_id_;
  uint32_t stream_id_;
  uint64_t timestamp_;
  std::string io_;
  size_t slot_;

 public:
  TensorStatDump(const std::string &original_kernel_name, const std::string &op_type, const std::string &op_name,
                 uint32_t task_id, uint32_t stream_id, uint64_t timestamp, bool input, size_t slot);
  void DumpTensorStatsToFile(const std::string &dump_path, const Debugger *debugger);
};
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_DEBUG_DATA_DUMP_TENSOR_STAT_DUMP_H_
