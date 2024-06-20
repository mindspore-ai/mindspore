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
#include "debug/debugger/debugger_utils.h"
#include "include/backend/debug/common/csv_writer.h"
#include "include/backend/debug/data_dump/dump_utils.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/backend/debug/data_dump/dump_json_parser.h"
#include "debug/utils.h"

namespace mindspore {

namespace {
using TensorPtr = tensor::TensorPtr;

constexpr auto kInput = "input";
constexpr auto kOutput = "output";
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

TensorStat GetKernelTensorStats(const DumpTensorInfo &tensor_info, const std::vector<string> &stat_name_list) {
  auto tensor = tensor_info.tensor;
  if (tensor == nullptr) {
    MS_LOG(WARNING) << "Tensor is nullptr, returning empty tensor statistics.";
    return TensorStat();
  }

  const auto &shape_vec = tensor->GetShapeVector();
  string shape = ShapeToString(shape_vec);
  size_t data_count = SizeOf(shape_vec);
  size_t data_size = tensor->size();
  string data_type = TypeIdToString(tensor->dtype_id(), true);
  MS_LOG(DEBUG) << "Tensor shape is " << shape << ", size is " << data_size << ", type is " << data_type;
  auto is_calc_stat = [&stat_name_list](std::string name) {
    return (std::find(stat_name_list.begin(), stat_name_list.end(), name) != stat_name_list.end());
  };
  std::string max_value =
    is_calc_stat("max") ? TensorToString(CalStatistic("max", tensor_info.device_context, tensor)) : "0";
  std::string min_value =
    is_calc_stat("min") ? TensorToString(CalStatistic("min", tensor_info.device_context, tensor)) : "0";
  std::string mean_value =
    is_calc_stat("avg") ? TensorToString(CalStatistic("avg", tensor_info.device_context, tensor)) : "0";
  std::string norm_value =
    is_calc_stat("l2norm") ? TensorToString(CalStatistic("l2norm", tensor_info.device_context, tensor)) : "0";

  size_t task_id = 0;  // Under the kbyk, there is no concept of task_id. The default setting is 0.
  uint64_t timestamp = Common::GetTimeStamp();
  auto stream_id = tensor->stream_id();
  string io = (tensor_info.is_input ? kInput : kOutput);
  TensorStat stat(tensor_info.op_type, tensor_info.op_name, task_id, stream_id, timestamp, io, tensor_info.slot,
                  data_size, data_type, shape, max_value, min_value, mean_value, norm_value, data_count);
  return stat;
}

void DumpKernelTensorStats(const DeviceContext *device_context, vector<device::DeviceAddress *> tensors, bool is_input,
                           const CNodePtr &node, uint32_t graph_id) {
  string node_name = GetKernelNodeName(node);
  GetFileKernelName(NOT_NULL(&node_name));
  string node_type = common::AnfAlgo::GetCNodeName(node);
  MS_LOG(DEBUG) << "Start calc " << node_name << " node statistics.";
  const string csv_header = CsvHeaderUtil::GetInstance().GetStatCsvHeader();
  const std::vector<string> &stat_name_list = DumpJsonParser::GetInstance().statistic_category();
  uint32_t rank_id = GetRankId();
  string filename = GenerateDumpPath(graph_id, rank_id) + "/" + kCsvFileName;
  CsvWriter csv;
  if (!csv.OpenFile(filename, csv_header)) {
    MS_LOG(WARNING) << "filename is " << filename;
    MS_LOG(WARNING) << "Open statistic dump file failed, skipping current statistics";
    return;
  }
  auto valid_index = GetValidDumpIndex(node, tensors.size(), is_input);
  for (auto i : valid_index) {
    auto tensor = tensors[i]->kernel_tensor().get();
    DumpTensorInfo tensor_info(device_context, tensor, is_input, i, node_name, node_type);
    auto stat = GetKernelTensorStats(tensor_info, stat_name_list);
    stat.UpdateHeaderItemMap();

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

    for (const auto &name : stat_name_list) {
      // DumpJsonParse guarantee names are valid.
      auto stat_val = stat.header_item_map[name];
      csv.WriteToCsv(stat_val);
    }
    csv.WriteToCsv("", true);
  }
  csv.CloseFile();
}

}  // namespace datadump
}  // namespace mindspore
