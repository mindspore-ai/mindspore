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

#include "plugin/device/ascend/hal/profiler/memory_profiling.h"
#include <fstream>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "nlohmann/json.hpp"
#include "plugin/device/ascend/hal/profiler/options.h"

namespace mindspore {
namespace profiler {
namespace ascend {
constexpr char kOutputPath[] = "output";

void MemoryProfiling::SetMemoryProfilingInitialize(const std::string &profiling_options) {
  nlohmann::json options;
  try {
    options = nlohmann::json::parse(profiling_options);
  } catch (nlohmann::json::exception &e) {
    MS_LOG(EXCEPTION) << "Failed to parse profiling options because of format error.";
  }

  if (options["profile_memory"] == "on") {
    is_initialized_ = true;
  }
}

void MemoryProfiling::StartMemoryProfiling() {
  is_enabled_ = true;
  if (NeedSaveMemoryProfiling()) {
    SaveMemoryProfiling();
    has_save_memory_data_ = true;
  }
}

void MemoryProfiling::StopMemoryProfiling() { is_enabled_ = false; }

std::shared_ptr<GraphMemory> MemoryProfiling::AddGraphMemoryNode(uint32_t graph_id) {
  std::shared_ptr<GraphMemory> node = std::make_shared<GraphMemory>(graph_id);
  MS_EXCEPTION_IF_NULL(node);
  graph_memory_[graph_id] = node;
  return node;
}

std::shared_ptr<GraphMemory> MemoryProfiling::GetGraphMemoryNode(uint32_t graph_id) const {
  auto node = graph_memory_.find(graph_id);
  if (node != graph_memory_.end()) {
    return node->second;
  }

  return nullptr;
}

bool MemoryProfiling::MemoryToPB() {
  memory_proto_.set_total_mem(device_mem_size_);
  if (graph_memory_.size() == 0) {
    MS_LOG(INFO) << "No memory profiling data need to be reported.";
    return false;
  }

  for (const auto &graph : graph_memory_) {
    GraphMemProto *graph_proto = memory_proto_.add_graph_mem();
    if (graph_proto == nullptr) {
      MS_LOG(ERROR) << "Add graph memory proto failed.";
      return false;
    }
    graph_proto->set_graph_id(graph.second->GetGraphId());
    graph_proto->set_static_mem(graph.second->GetStaticMemSize());
    // node memory to PB
    for (const auto &node : graph.second->GetNodeMemory()) {
      NodeMemProto *node_mem = graph_proto->add_node_mems();
      if (node_mem == nullptr) {
        MS_LOG(ERROR) << "Add node memory proto failed.";
        return false;
      }
      node_mem->set_node_name(node.GetNodeName());
      node_mem->set_node_id(node.GetNodeId());
      for (const auto &id : node.GetInputTensorId()) {
        node_mem->add_input_tensor_id(id);
      }
      for (const auto &id : node.GetOutputTensorId()) {
        node_mem->add_output_tensor_id(id);
      }
      for (const auto &id : node.GetOutputTensorId()) {
        node_mem->add_workspace_tensor_id(id);
      }
    }
    // tensor memory to PB
    for (const auto &node : graph.second->GetTensorMemory()) {
      TensorMemProto *tensor_mem = graph_proto->add_tensor_mems();
      if (tensor_mem == nullptr) {
        MS_LOG(ERROR) << "Add node memory proto failed.";
        return false;
      }
      tensor_mem->set_tensor_id(node.GetTensorId());
      tensor_mem->set_size(node.GetAlignedSize());
      std::string type = node.GetType();
      tensor_mem->set_type(type);
      tensor_mem->set_life_start(node.GetLifeStart());
      tensor_mem->set_life_end(node.GetLifeEnd());
      std::string life_long = node.GetLifeLong();
      tensor_mem->set_life_long(life_long);
    }
  }
  MS_LOG(INFO) << "Memory profiling data to PB end.";
  return true;
}

void MemoryProfiling::SaveMemoryProfiling() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string dir_path = GetOutputPath();
  auto device_id = common::GetEnv("RANK_ID");
  // If RANK_ID is not set, default value is 0
  if (device_id.empty()) {
    device_id = "0";
  }

  if (!MemoryToPB()) {
    return;
  }

  std::string file = dir_path + std::string("/memory_usage_") + std::string(device_id) + std::string(".pb");
  std::fstream handle(file, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!memory_proto_.SerializeToOstream(&handle)) {
    MS_LOG(ERROR) << "Save memory profiling data to file failed";
  }
  handle.close();

  MS_LOG(INFO) << "Start save memory profiling data to " << file << " end";
  return;
}
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
