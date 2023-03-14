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

#include "include/backend/debug/data_dump/tensor_stat_dump.h"

#include <memory>
#include <map>
#include "include/common/debug/common.h"
#include "debug/debug_services.h"
#include "include/backend/debug/debugger/debugger.h"
#include "debug/data_dump/csv_writer.h"

namespace {
constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kCsvHeader =
  "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape,Max Value,Min Value,Avg Value,"
  "Count,Negative Zero Count,Positive Zero Count,NaN Count,Negative Inf Count,Positive Inf Count,Zero Count\n";
constexpr auto kCsvFileName = "statistic.csv";
}  // namespace

namespace mindspore {
TensorStatDump::TensorStatDump(const std::string &op_type, const std::string &op_name, uint32_t task_id,
                               uint32_t stream_id, uint64_t timestamp, bool input, size_t slot,
                               size_t tensor_loader_slot)
    : op_type_{op_type},
      op_name_{op_name},
      task_id_{std::to_string(task_id)},
      stream_id_{std::to_string(stream_id)},
      timestamp_{std::to_string(timestamp)},
      slot_{slot},
      tensor_loader_slot_{tensor_loader_slot} {
  if (input) {
    io_ = kInput;
  } else {
    io_ = kOutput;
  }
}

TensorStatDump::TensorStatDump(const std::string &op_type, const std::string &op_name, const std::string &task_id,
                               const std::string &stream_id, const std::string &timestamp, const std::string &io,
                               size_t slot, size_t tensor_loader_slot)
    : op_type_{op_type},
      op_name_{op_name},
      task_id_{task_id},
      stream_id_{stream_id},
      timestamp_{timestamp},
      io_{io},
      slot_{slot},
      tensor_loader_slot_{tensor_loader_slot} {
  if (io_ != kInput && io_ != kOutput) {
    MS_LOG(EXCEPTION) << "Cannot instantiate TensorStatDump, io needs to be either " << kInput << " or " << kOutput;
  }
}

bool TensorStatDump::OpenStatisticsFile(const std::string &dump_path) {
  std::string filename = dump_path + "/" + kCsvFileName;
  // try to open file
  CsvWriter &csv = CsvWriter::GetInstance();
  int retry = 2;
  while (retry > 0) {
    if (csv.OpenFile(filename, kCsvHeader)) {
      break;
    }
    retry--;
  }
  if (retry == 0) {
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return false;
  }
  return true;
}

bool TensorStatDump::DumpTensorStatsToFile(const std::string &original_kernel_name, const std::string &dump_path,
                                           const Debugger *debugger) {
  // get tensor data using debugger
  std::string tensor_loader_name = original_kernel_name + ":" + std::to_string(tensor_loader_slot_);
  std::shared_ptr<TensorData> data = debugger->GetTensor(tensor_loader_name);
  if (data == nullptr) {
    MS_LOG(INFO) << "Failed to find " << tensor_loader_name << " in tensor loader, skipping current statistics";
    return false;
  }
  return DumpTensorStatsToFile(dump_path, data);
}

bool TensorStatDump::DumpTensorStatsToFile(const std::string &dump_path, const std::shared_ptr<TensorData> data) {
  if (data == nullptr) {
    MS_LOG(INFO) << "Tensor data is empty, skipping current statistics";
    return false;
  }
  std::string type = data->GetTypeString();
  if (type.empty()) {
    type = "unsupported(" + std::to_string(data->GetType()) + ")";
    MS_LOG(INFO) << "Unsupported tensor data_type " << type << " for tensor " << data->GetName();
  }
  if (!OpenStatisticsFile(dump_path)) {
    return false;
  }
  const DebugServices::TensorStat &stat = DebugServices::GetTensorStatistics(data);
  // write tensor statistics to csv file
  std::ostringstream shape;
  shape << "\"(";
  for (size_t i = 0; i < stat.shape.size(); i++) {
    shape << (i > 0 ? "," : "") << stat.shape[i];
  }
  shape << ")\"";
  CsvWriter &csv = CsvWriter::GetInstance();
  csv.WriteToCsv(op_type_);
  csv.WriteToCsv(op_name_);
  csv.WriteToCsv(task_id_);
  csv.WriteToCsv(stream_id_);
  csv.WriteToCsv(timestamp_);
  csv.WriteToCsv(io_);
  csv.WriteToCsv(slot_);
  csv.WriteToCsv(stat.data_size);
  csv.WriteToCsv(type);
  csv.WriteToCsv(shape.str());
  if (stat.count == stat.nan_count + stat.neg_inf_count + stat.pos_inf_count) {
    csv.WriteToCsv(std::string("null"));
    csv.WriteToCsv(std::string("null"));
    csv.WriteToCsv(std::string("null"));
  } else {
    csv.WriteToCsv(stat.max_value);
    csv.WriteToCsv(stat.min_value);
    csv.WriteToCsv(stat.avg_value);
  }
  csv.WriteToCsv(stat.count);
  csv.WriteToCsv(stat.neg_zero_count);
  csv.WriteToCsv(stat.pos_zero_count);
  csv.WriteToCsv(stat.nan_count);
  csv.WriteToCsv(stat.neg_inf_count);
  csv.WriteToCsv(stat.pos_inf_count);
  csv.WriteToCsv(stat.zero_count, true);
  return true;
}
}  // namespace mindspore
