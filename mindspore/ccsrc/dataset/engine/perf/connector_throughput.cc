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

#include <fstream>
#include <iterator>
#include <algorithm>
#include <memory>
#include <string>
#include <nlohmann/json.hpp>
#include "dataset/engine/perf/connector_throughput.h"
#include "dataset/engine/execution_tree.h"
#include "dataset/util/path.h"

namespace mindspore {
namespace dataset {

// temporary helper
int ConnectorThroughput::InitNodes() {
  auto it = (*tree_).begin();
  return it.NumNodes();
}
// Sample action
Status ConnectorThroughput::Sample() {
  std::vector<int64_t> out_buffer_count_row(n_nodes_);
  std::vector<double> throughput_row(n_nodes_);
  TimePoint cur_time;  // initialised inside the loop, used outside the loop to update prev sample time.
  auto col = 0;
  for (const auto &node : *tree_) {
    auto cur_out_buffer_count = node.ConnectorOutBufferCount();
    out_buffer_count_row[col] = cur_out_buffer_count;
    auto sz = timestamps_.size();
    cur_time = std::chrono::steady_clock::now();
    auto _dt = std::chrono::duration_cast<std::chrono::microseconds>(timestamps_[0][sz - 1] - timestamps_[0][sz - 2]);
    auto dt = std::chrono::duration<double>(_dt).count();
    auto prev_out_buffer_count = out_buffer_count_table_[col][out_buffer_count_table_.size() - 1];
    if (dt != 0) {
      auto thr = (cur_out_buffer_count - prev_out_buffer_count) / (1000 * dt);
      throughput_row[col] = thr;
    } else {
      throughput_row[col] = -1;
    }
    col++;
  }
  std::vector<TimePoint> v = {cur_time};  // temporary fix
  timestamps_.AddSample(v);
  // Push new row of sample
  out_buffer_count_table_.AddSample(out_buffer_count_row);
  throughput_.AddSample(throughput_row);
  return Status::OK();
}

json ConnectorThroughput::ParseOpInfo(const DatasetOp &node, const std::vector<double> &thr) {
  auto children = node.Children();
  std::vector<int32_t> children_id;
  std::transform(children.begin(), children.end(), std::back_inserter(children_id),
                 [](std::shared_ptr<DatasetOp> op) -> int32_t { return op->id(); });
  json json_node;
  json_node["op_id"] = node.id();
  json_node["op_type"] = node.Name();
  json_node["num_workers"] = node.num_workers();
  json metrics;
  metrics["output_queue"] = {{"throughput", thr}};

  json_node["metrics"] = metrics;
  if (!children_id.empty()) {
    json_node["children"] = children_id;
  }

  return json_node;
}

// Save profiling data to file
Status ConnectorThroughput::SaveToFile() {
  std::ofstream os(file_path_);
  json output;
  output["sampling_interval"] = 10;
  // Traverse the ExecutionTree for JSON node generation
  int col = 0;
  for (auto &node : *tree_) {
    std::vector<double> throughput;
    for (auto i = 0; i < throughput_.size(); i++) {
      throughput.push_back(throughput_[col][i]);
    }
    json json_node = ParseOpInfo(node, throughput);
    output["op_info"].push_back(json_node);
    col++;
  }
  os << output;
  return Status::OK();
}
Status ConnectorThroughput::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("pipeline_profiling_" + Name() + "_" + device_id + ".json")).toString();
  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
