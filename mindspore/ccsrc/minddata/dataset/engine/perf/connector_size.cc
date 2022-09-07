/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "minddata/dataset/engine/perf/connector_size.h"

#include <algorithm>
#include <fstream>
#include <memory>

#include "minddata/dataset/core/config_manager.h"
#include "minddata/dataset/engine/execution_tree.h"
#include "minddata/dataset/util/path.h"

using json = nlohmann::json;
namespace mindspore {
namespace dataset {
using Qrow = std::vector<int>;

// Sample action
Status ConnectorSize::Sample() {
  if (!active_) {
    return Status::OK();
  }
  Qrow cur_row;
  (void)std::transform(tree_->begin(), tree_->end(), std::back_inserter(cur_row),
                       [](const DatasetOp &op) { return op.ConnectorSize(); });
  // Tree Iterator is in PostOrder (leaf first, e.g., 3,2,1)
  // reverse the order of the vector to get the root first.
  std::reverse(cur_row.begin(), cur_row.end());
  std::lock_guard<std::mutex> guard(lock_);
  // Push new row of sample
  sample_table_.push_back(cur_row);
  (void)ts_.emplace_back(ProfilingTime::GetCurMilliSecond());
  return Status::OK();
}

// JSON serializer helper function
json ConnectorSize::ParseOpInfo(const DatasetOp &node) const {
  json json_node;
  json_node["op_id"] = node.id();
  json_node["op_type"] = node.Name();
  json_node["num_workers"] = node.NumWorkers();
  json metrics;
  // DataQueueOp is a special op,it is not inlined but its output queue is invalid.
  // So we should not output its queue size.
  if (!node.inlined() && node.Name() != "DataQueueOp") {
    metrics["output_queue"] = {{"length", node.ConnectorCapacity()}};
  }
  json_node["metrics"] = metrics;

  auto children = node.Children();
  std::vector<int32_t> children_id;
  (void)std::transform(children.begin(), children.end(), std::back_inserter(children_id),
                       [](const std::shared_ptr<DatasetOp> &op) -> int32_t { return op->id(); });
  if (!children_id.empty()) {
    json_node["children"] = children_id;
  }

  return json_node;
}

// Save profiling data to file
Status ConnectorSize::SaveToFile(const std::string &dir_path, const std::string &rank_id) {
  Path path = GetFileName(dir_path, rank_id);
  // Remove the file if it exists (from prior profiling usage)
  RETURN_IF_NOT_OK(path.Remove());
  std::string file_path = path.ToString();

  json output = initial_nodes_data;
  output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();

  // Traverse the JSON initialized in Init() to access each op's information
  CHECK_FAIL_RETURN_UNEXPECTED(output.contains("op_info"), "JSON data does not include op_info!");
  for (uint32_t idx = 0; idx < output["op_info"].size(); idx++) {
    std::vector<int32_t> cur_queue_size;
    (void)std::transform(sample_table_.begin(), sample_table_.end(), std::back_inserter(cur_queue_size),
                         [&](const ConnectorSizeSample &sample) { return sample[idx]; });

    auto &ops_data = output["op_info"];
    if (ops_data[idx]["metrics"].contains("output_queue") && ops_data[idx]["op_type"] != "DataQueueOp") {
      ops_data[idx]["metrics"]["output_queue"]["size"] = cur_queue_size;
    }
  }

  // Discard the content of the file when opening.
  std::ofstream os(file_path, std::ios::trunc);
  os << output;
  os.close();
  return Status::OK();
}

Status ConnectorSize::Init() {
  // Traverse the ExecutionTree for JSON node generation
  for (auto &node : *tree_) {
    json json_node = ParseOpInfo(node);
    initial_nodes_data["op_info"].push_back(json_node);
  }
  // Tree Iterator is in PostOrder (leaf first, e.g., 3,2,1)
  // reverse the order of the vector to get the root first.
  std::reverse(initial_nodes_data["op_info"].begin(), initial_nodes_data["op_info"].end());

  return Status::OK();
}

void ConnectorSize::Clear() {
  ts_.clear();
  sample_table_.clear();
  initial_nodes_data.clear();
}

Status ConnectorSize::GetOpConnectorSize(int32_t op_id, uint64_t start_time, uint64_t end_time,
                                         std::vector<int32_t> *result) {
  RETURN_UNEXPECTED_IF_NULL(result);
  MS_LOG(DEBUG) << "Op_id: " << op_id << " start_ts: " << start_time << " end_ts: " << end_time;
  CHECK_FAIL_RETURN_UNEXPECTED(start_time < end_time,
                               "Expected start_time < end_time. Got start_ts: " + std::to_string(start_time) +
                                 " end_ts: " + std::to_string(end_time));
  std::lock_guard<std::mutex> guard(lock_);
  CHECK_FAIL_RETURN_UNEXPECTED(
    ts_.size() == sample_table_.size(),
    "Expected ts_.size() == sample_table_.size(). Got ts_.size: " + std::to_string(ts_.size()) +
      " sample_table_.size: " + std::to_string(sample_table_.size()));
  // find first ts that is not less than start_ts
  auto lower = std::lower_bound(ts_.begin(), ts_.end(), start_time);
  // find first ts that is greater than end_ts
  auto upper = std::upper_bound(ts_.begin(), ts_.end(), end_time);
  // get ts_ indices
  auto start_index = std::distance(ts_.begin(), lower);
  auto end_index = std::distance(ts_.begin(), upper);
  MS_LOG(INFO) << "start_index: " << start_index << " end_index: " << end_index;
  CHECK_FAIL_RETURN_UNEXPECTED(start_index <= end_index,
                               "Expected start_index <= end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  // convert indices to sample_table_ iterator
  auto first_iter = sample_table_.begin() + start_index;
  auto last_iter = sample_table_.begin() + end_index;
  // op_id corresponds to the index in sample vector
  (void)std::transform(first_iter, last_iter, std::back_inserter(*result),
                       [&](const ConnectorSizeSample &sample) { return sample[static_cast<size_t>(op_id)]; });

  return Status::OK();
}

Path ConnectorSize::GetFileName(const std::string &dir_path, const std::string &rank_id) {
  return Path(dir_path) / Path("pipeline_profiling_" + rank_id + ".json");
}
}  // namespace dataset
}  // namespace mindspore
