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

#include "debug/data_dump/tensor_stat_dump.h"

#include <memory>
#include "utils/file_utils.h"
#include "debug/common.h"
#include "debug/debug_services.h"
#include "debug/debugger/debugger.h"

namespace {
constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kCsvHeader =
  "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape,Max Value,Min Value,Avg Value,"
  "Count,Negative Zero Count,Positive Zero Count,NaN Count,Negative Inf Count,Positive Inf Count,Zero Count\n";
constexpr auto kCsvFileName = "statistic.csv";
}  // namespace
namespace mindspore {
bool CsvWriter::OpenFile(const std::string &path, const std::string &header) {
  if (file_.is_open() && path == file_path_str_) {
    return true;
  }
  if (file_.is_open()) {
    CloseFile();
  }
  bool first_time_opening = file_path_str_ != path;
  ChangeFileMode(path, S_IWUSR);
  if (first_time_opening) {
    // remove any possible output from previous runs
    file_.open(path, std::ios::out | std::ios::trunc | std::ios::binary);
  } else {
    file_.open(path, std::ios::out | std::ios::app | std::ios::binary);
  }
  if (!file_.is_open()) {
    MS_LOG(WARNING) << "Open file " << path << " failed." << ErrnoToString(errno);
    return false;
  }
  if (first_time_opening) {
    file_ << header;
    file_.flush();
    file_path_str_ = path;
  }
  MS_LOG(INFO) << "Opened statistics file: " << path;
  return true;
}

void CsvWriter::CloseFile() {
  if (file_.is_open()) {
    file_.close();
    ChangeFileMode(file_path_str_, S_IRUSR);
    MS_LOG(INFO) << "Closed statistics dump file: " << file_path_str_;
  }
}

template <typename T>
void CsvWriter::WriteToCsv(const T &val, bool end_line) {
  file_ << val;
  if (end_line) {
    file_ << kEndLine;
    file_.flush();
  } else {
    file_ << kSeparator;
  }
}

CsvWriter::~CsvWriter() { CloseFile(); }

TensorStatDump::TensorStatDump(const std::string &original_kernel_name, const std::string &op_type,
                               const std::string &op_name, uint32_t task_id, uint32_t stream_id, uint64_t timestamp,
                               bool input, size_t slot)
    : original_kernel_name_{original_kernel_name},
      op_type_{op_type},
      op_name_{op_name},
      task_id_{task_id},
      stream_id_{stream_id},
      timestamp_{timestamp},
      slot_{slot} {
  if (input) {
    io_ = kInput;
  } else {
    io_ = kOutput;
  }
}

void TensorStatDump::DumpTensorStatsToFile(const std::string &dump_path, const Debugger *debugger) {
  std::string filename = dump_path + "/" + kCsvFileName;
  auto file_path = Common::CreatePrefixPath(filename);
  if (!file_path.has_value()) {
    MS_LOG(WARNING) << "CreatePrefixPath failed.";
    return;
  }
  // try to open file
  CsvWriter &csv = CsvWriter::GetInstance();
  std::string file_path_value = file_path.value();
  int retry = 2;
  while (retry > 0) {
    if (csv.OpenFile(file_path_value, kCsvHeader)) {
      break;
    }
    retry--;
  }
  if (!retry) {
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return;
  }
  // get tensor statistics using debugger
  std::string tensor_loader_name = original_kernel_name_ + ":" + std::to_string(slot_);
  std::shared_ptr<TensorData> data = debugger->GetTensor(tensor_loader_name);
  if (data == nullptr) {
    MS_LOG(WARNING) << "Failed to find tensor in tensor loader, skipping current statistics";
    return;
  }
  const DebugServices::TensorStat &stat = debugger->GetTensorStatistics(data);
  // write tensor statistics to csv file
  std::ostringstream shape;
  shape << "\"(";
  for (size_t i = 0; i < stat.shape.size(); i++) {
    shape << (i ? "," : "") << stat.shape[i];
  }
  shape << ")\"";
  csv.WriteToCsv(op_type_);
  csv.WriteToCsv(op_name_);
  csv.WriteToCsv(task_id_);
  csv.WriteToCsv(stream_id_);
  csv.WriteToCsv(timestamp_);
  csv.WriteToCsv(io_);
  csv.WriteToCsv(slot_);
  csv.WriteToCsv(stat.data_size);
  csv.WriteToCsv(stat.dtype);
  csv.WriteToCsv(shape.str());
  csv.WriteToCsv(stat.max_value);
  csv.WriteToCsv(stat.min_value);
  csv.WriteToCsv(stat.avg_value);
  csv.WriteToCsv(stat.count);
  csv.WriteToCsv(stat.neg_zero_count);
  csv.WriteToCsv(stat.pos_zero_count);
  csv.WriteToCsv(stat.nan_count);
  csv.WriteToCsv(stat.neg_inf_count);
  csv.WriteToCsv(stat.pos_inf_count);
  csv.WriteToCsv(stat.zero_count, true);
}
}  // namespace mindspore
