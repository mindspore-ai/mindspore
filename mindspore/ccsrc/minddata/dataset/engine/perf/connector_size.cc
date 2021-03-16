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
#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
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
  std::transform(tree_->begin(), tree_->end(), std::back_inserter(cur_row),
                 [](DatasetOp &op) { return op.ConnectorSize(); });
  // Push new row of sample
  sample_table_.push_back(cur_row);
  return Status::OK();
}

// JSON serializer helper function
json ConnectorSize::ParseOpInfo(const DatasetOp &node, const std::vector<int32_t> &size) {
  auto children = node.Children();
  std::vector<int32_t> children_id;
  std::transform(children.begin(), children.end(), std::back_inserter(children_id),
                 [](std::shared_ptr<DatasetOp> op) -> int32_t { return op->id(); });
  json json_node;
  json_node["op_id"] = node.id();
  json_node["op_type"] = node.Name();
  json_node["num_workers"] = node.num_workers();
  json metrics;
  // DeviceQueueOp is a special op,it is not inlined but its output queue is invalid.
  // So we should not output its queue size.
  if (!node.inlined() && node.Name() != "DeviceQueueOp") {
    metrics["output_queue"] = {{"size", size}, {"length", node.ConnectorCapacity()}};
  }
  json_node["metrics"] = metrics;
  if (!children_id.empty()) {
    json_node["children"] = children_id;
  }

  return json_node;
}

// Save profiling data to file
// If the file is already exist (created by other sampling node), simply add the data to metrics field.
Status ConnectorSize::SaveToFile() {
  Path path = Path(file_path_);
  json output;
  if (path.Exists()) {
    MS_LOG(DEBUG) << file_path_ << " exists";
    try {
      std::ifstream file(file_path_);
      file >> output;
    } catch (const std::exception &err) {
      RETURN_STATUS_UNEXPECTED("Invalid file, failed to open json file: " + file_path_ +
                               ", please delete it and try again!");
    }
  } else {
    output["sampling_interval"] = GlobalContext::config_manager()->monitor_sampling_interval();
  }

  uint32_t idx = 0;
  // Traverse the ExecutionTree for JSON node generation
  for (auto &node : *tree_) {
    std::vector<int32_t> cur_queue_size;
    std::transform(sample_table_.begin(), sample_table_.end(), std::back_inserter(cur_queue_size),
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
  return Status::OK();
}

Status ConnectorSize::Init(const std::string &dir_path, const std::string &device_id) {
  file_path_ = (Path(dir_path) / Path("pipeline_profiling_" + device_id + ".json")).toString();
  return Status::OK();
}

Status ConnectorSize::Analyze() { return Status::OK(); }
}  // namespace dataset
}  // namespace mindspore
