/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "debug/data_dump/tensor_statistic.h"
#include <memory>
#include <string>
#include <vector>
#include "debug/data_dump/statistic_kernel.h"
#include "include/backend/debug/common/csv_writer.h"
#include "include/backend/debug/data_dump/dump_utils.h"

namespace mindspore {

namespace {
using TensorPtr = tensor::TensorPtr;

constexpr auto kInput = "input";
constexpr auto kOutput = "output";
constexpr auto kCsvHeader =
  "Op Type,Op Name,Task ID,Stream ID,Timestamp,IO,Slot,Data Size,Data Type,Shape,Max Value,Min Value,Avg Value,"
  "Count\n";
constexpr auto kCsvFileName = "statistic.csv";
string ShapeToString(const ShapeVector &shape) {
  std::ostringstream sstr;
  sstr << "\"(";
  for (size_t i = 0; i < shape.size(); i++) {
    sstr << (i > 0 ? "," : "") << shape[i];
  }
  sstr << ")\"";
  return string{sstr.str()};
}
string TensorToString(TensorPtr tensor) {
  if (!tensor) {
    return "null";
  }
  return tensor->data().ToString(tensor->data_type(), tensor->shape(), false);
}
}  // namespace

namespace datadump {

TensorStat GetKernelTensorStats(const DumpTensorInfo &tensor_info) {
  auto tensor = tensor_info.tensor;
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "Tensor is nullptr, returning empty tensor statistics.";
    return TensorStat();
  }
  uint64_t timestamp = Common::GetTimeStamp();
  std::string max_value =
    TensorToString(StatisticKernelManager::GetInstance().CalMax(tensor_info.device_context, tensor));
  std::string min_value =
    TensorToString(StatisticKernelManager::GetInstance().CalMin(tensor_info.device_context, tensor));
  std::string mean_value =
    TensorToString(StatisticKernelManager::GetInstance().CalMean(tensor_info.device_context, tensor));
  const auto &shape_vec = tensor->GetShapeVector();
  size_t task_id = 0;  // Under the kbyk, there is no concept of task_id. The default setting is 0.
  string shape = ShapeToString(shape_vec);

  auto stream_id = tensor->stream_id();
  size_t data_size = tensor->size();
  string data_type = TypeIdToString(tensor->dtype_id());
  string io = (tensor_info.is_input ? kInput : kOutput);
  size_t data_count = SizeOf(shape_vec);
  TensorStat stat(tensor_info.op_type, tensor_info.op_name, task_id, stream_id, timestamp, io, tensor_info.slot,
                  data_size, data_type, shape, max_value, min_value, mean_value, data_count);
  return stat;
}

void DumpKernelTensorStats(const DeviceContext *device_context, vector<KernelTensor *> tensors, bool is_input,
                           const string &op_name, const string &op_type) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    DumpTensorInfo tensor_info(device_context, tensors[i], is_input, i, op_name, op_type);
    auto stat = GetKernelTensorStats(tensor_info);
    uint32_t rank_id = GetRankId();
    string filename = GenerateDumpPath(0, rank_id) + "/" + kCsvFileName;
    CsvWriter csv;
    if (!csv.OpenFile(filename, kCsvHeader)) {
      MS_LOG(WARNING) << "filename is " << filename;
      MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
      return;
    }
    csv.WriteToCsv(stat.type_);
    csv.WriteToCsv(stat.name_);
    csv.WriteToCsv(stat.task_id_);
    csv.WriteToCsv(stat.stream_id_);
    csv.WriteToCsv(stat.timestamp_);
    csv.WriteToCsv(stat.io_);
    csv.WriteToCsv(stat.slot_);
    csv.WriteToCsv(stat.data_size_);
    csv.WriteToCsv(stat.data_type_);
    csv.WriteToCsv(stat.shape_);
    csv.WriteToCsv(stat.max_value_);
    csv.WriteToCsv(stat.min_value_);
    csv.WriteToCsv(stat.avg_value_);
    csv.WriteToCsv(stat.count_, true);
    csv.CloseFile();
  }
}

}  // namespace datadump
}  // namespace mindspore
