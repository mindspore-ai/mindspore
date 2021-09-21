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
#include "minddata/dataset/engine/perf/connector_size.h"
#include <fstream>
#include <algorithm>
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
  Qrow cur_row;
  (void)std::transform(tree_->begin(), tree_->end(), std::back_inserter(cur_row),
                       [](DatasetOp &op) { return op.ConnectorSize(); });
  std::lock_guard<std::mutex> guard(lock_);
  // Push new row of sample
  sample_table_.push_back(cur_row);
  (void)ts_.emplace_back(ProfilingTime::GetCurMilliSecond());
  return Status::OK();
}

// JSON serializer helper function
json ConnectorSize::ParseOpInfo(const DatasetOp &node, const std::vector<int32_t> &size) {
  json json_node;
  json_node["op_id"] = node.id();
  json_node["op_type"] = node.Name();
  json_node["num_workers"] = node.NumWorkers();
  json metrics;
  // DeviceQueueOp is a special op,it is not inlined but its output queue is invalid.
  // So we should not output its queue size.
  if (!node.inlined() && node.Name() != "DeviceQueueOp") {
    metrics["output_queue"] = {{"size", size}, {"length", node.ConnectorCapacity()}};
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
// If the file is already exist (created by other sampling node), simply add the data to metrics field.
Status ConnectorSize::SaveToFile() {
  json output;
  RETURN_IF_NOT_OK(ReadJson(&output));

  Path path = Path(file_path_);
  uint32_t idx = 0;
  // Traverse the ExecutionTree for JSON node generation
  for (auto &node : *tree_) {
    std::vector<int32_t> cur_queue_size;
    (void)std::transform(sample_table_.begin(), sample_table_.end(), std::back_inserter(cur_queue_size),
                         [&](const ConnectorSizeSample &sample) { return sample[idx]; });
    if (!path.Exists()) {
      json json_node = ParseOpInfo(node, cur_queue_size);
      output["op_info"].push_back(json_node);
    } else {
      if (!node.inlined() && node.Name() != "DeviceQueueOp") {
        auto &ops_data = output["op_info"];
        ops_data[idx]["metrics"]["output_queue"]["size"] = cur_queue_size;
        ops_data[idx]["metrics"]["output_queue"]["length"] = node.ConnectorCapacity();
      }
    }

    idx++;
  }

  // Discard the content of the file when opening.
  std::ofstream os(file_path_, std::ios::trunc);
  os << output;
  os.close();
  return Status::OK();
}

Status ConnectorSize::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("pipeline_profiling_" + device_id + ".json")).ToString();
  Path path = Path(file_path_);
  // Remove the file if it exists (from prior profiling usage)
  RETURN_IF_NOT_OK(path.Remove());
  return Status::OK();
}

Status ConnectorSize::Analyze() { return Status::OK(); }

Status ConnectorSize::GetOpConnectorSize(int32_t op_id, uint64_t start_time, uint64_t end_time,
                                         std::vector<int32_t> *result) {
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
  CHECK_FAIL_RETURN_UNEXPECTED(start_index < end_index,
                               "Expected start_index < end_index. Got start_index: " + std::to_string(start_index) +
                                 " end_index: " + std::to_string(end_index));
  // convert indices to sample_table_ iterator
  auto first_iter = sample_table_.begin() + start_index;
  auto last_iter = sample_table_.begin() + end_index;
  // op_id corresponds to the index in sample vector
  (void)std::transform(first_iter, last_iter, std::back_inserter(*result),
                       [&](const ConnectorSizeSample &sample) { return sample[op_id]; });

  return Status::OK();
}
}  // namespace dataset
}  // namespace mindspore
