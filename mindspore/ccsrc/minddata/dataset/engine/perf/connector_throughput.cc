/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <sys/stat.h>
#include <iterator>
#include <algorithm>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "minddata/dataset/engine/perf/connector_throughput.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"

namespace mindspore {
namespace dataset {

// temporary helper
int ConnectorThroughput::InitNodes() {
  if (tree_ == nullptr) {
    return 0;
  }
  auto it = (*tree_).begin();
  return it.NumNodes();
}
// Sample action
Status ConnectorThroughput::Sample() {
  std::vector<int64_t> out_row_count_row(n_nodes_);
  std::vector<double> throughput_row(n_nodes_);
  TimePoint cur_time;  // initialised inside the loop, used outside the loop to update prev sample time.
  auto col = 0;
  for (const auto &node : *tree_) {
    auto cur_out_rows_count = node.ConnectorOutRowsCount();
    out_row_count_row[col] = cur_out_rows_count;
    auto sz = timestamps_.size();
    cur_time = std::chrono::steady_clock::now();
    double data_time = 0;
    if (sz > 1) {
      auto full_time =
        std::chrono::duration_cast<std::chrono::microseconds>(timestamps_[0][sz - 1] - timestamps_[0][sz - 2]);
      data_time = std::chrono::duration<double>(full_time).count();
    }
    auto prev_out_rows_count = out_row_count_table_[col][out_row_count_table_.size() - 1];
    if (data_time != 0) {
      const int32_t multiplier = 1000;
      auto thr = (cur_out_rows_count - prev_out_rows_count) / (multiplier * data_time);
      throughput_row[col] = thr;
    } else {
      throughput_row[col] = 0;
    }
    col++;
  }
  std::vector<TimePoint> v = {cur_time};  // temporary fix
  timestamps_.AddSample(v);
  // Push new row of sample
  out_row_count_table_.AddSample(out_row_count_row);
  throughput_.AddSample(throughput_row);
  return Status::OK();
}

json ConnectorThroughput::ParseOpInfo(const DatasetOp &node, const std::vector<double> &thr) {
  auto children = node.Children();
  std::vector<int32_t> children_id;
  std::transform(children.begin(), children.end(), std::back_inserter(children_id),
                 [](const std::shared_ptr<DatasetOp> &op) -> int32_t { return op ? op->id() : 0; });
  json json_node;
  json_node["op_id"] = node.id();
  json_node["op_type"] = node.Name();
  json_node["num_workers"] = node.NumWorkers();
  json metrics;
  // DeviceQueueOp is a special op,it is not inlined but its output queue is invalid.
  // So we should not output its connector throughput.
  if (!node.inlined() && node.Name() != "DeviceQueueOp") {
    metrics["output_queue"] = {{"throughput", thr}};
  }
  json_node["metrics"] = metrics;
  if (!children_id.empty()) {
    json_node["children"] = children_id;
  }

  return json_node;
}

// Save profiling data to file
// If the file is already exist (created by other sampling node), simply add the data to metrics field.
Status ConnectorThroughput::SaveToFile() {
  json output;
  RETURN_IF_NOT_OK(ReadJson(&output));

  Path path = Path(file_path_);
  // Traverse the ExecutionTree for JSON node generation
  int col = 0;
  for (auto &node : *tree_) {
    std::vector<double> throughput;
    if (throughput_.size() > col) {
      for (auto i = 0; i < throughput_[col].size(); i++) {
        throughput.push_back(throughput_[col][i]);
      }
    }

    if (!path.Exists()) {
      json json_node = ParseOpInfo(node, throughput);
      output["op_info"].push_back(json_node);
    } else {
      if (!node.inlined() && node.Name() != "DeviceQueueOp") {
        auto &ops_data = output["op_info"];
        ops_data[col]["metrics"]["output_queue"]["throughput"] = throughput;
      }
    }
    col++;
  }

  // Discard the content of the file when opening.
  std::ofstream os(file_path_, std::ios::trunc);
  os << output;
  os.close();
  return Status::OK();
}

Status ConnectorThroughput::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("pipeline_profiling_" + device_id + ".json")).ToString();
  Path path = Path(file_path_);
  // Remove the file if it exists (from prior profiling usage)
  RETURN_IF_NOT_OK(path.Remove());
  return Status::OK();
}

Status ConnectorThroughput::ChangeFileMode() {
  if (file_path_.empty()) {
    return Status::OK();
  }

  if (chmod(common::SafeCStr(file_path_), S_IRUSR | S_IWUSR) == -1) {
    std::string err_str = "Change file mode failed," + file_path_;
    return Status(StatusCode::kMDUnexpectedError, err_str);
  }
  return Status::OK();
}

Status ConnectorThroughput::Analyze() { return Status::OK(); }
}  // namespace dataset
}  // namespace mindspore
