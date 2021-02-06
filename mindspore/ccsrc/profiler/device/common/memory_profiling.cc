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

#include "profiler/device/common/memory_profiling.h"
#include <fstream>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "nlohmann/json.hpp"

namespace mindspore {
namespace profiler {

constexpr char kOutputPath[] = "output";

std::shared_ptr<GraphMemory> MemoryProfiling::AddGraphMemoryNode(uint32_t graph_id) {
  std::shared_ptr<GraphMemory> node = std::make_shared<GraphMemory>(graph_id);
  graph_memory_[graph_id] = node;
  return node;
}

std::shared_ptr<GraphMemory> MemoryProfiling::GetGraphMemoryNode(uint32_t graph_id) {
  auto node = graph_memory_.find(graph_id);
  if (node != graph_memory_.end()) {
    return node->second;
  }

  return nullptr;
}

void MemoryProfiling::MemoryToPB() {
  memory_proto_.set_total_mem(device_mem_size_);
  for (const auto &graph : graph_memory_) {
    GraphMemProto *graph_proto = memory_proto_.add_graph_mem();
    graph_proto->set_graph_id(graph.second->GetGraphId());
    graph_proto->set_static_mem(graph.second->GetStaticMemSize());
    // node memory to PB
    for (const auto &node : graph.second->GetNodeMemory()) {
      NodeMemProto *node_mem = graph_proto->add_node_mems();
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
  MS_LOG(INFO) << "Memory profiling data to PB end";
  return;
}

std::string MemoryProfiling::GetOutputPath() const {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  const std::string options_str = context->get_param<std::string>(MS_CTX_PROFILING_OPTIONS);
  nlohmann::json options_json;
  try {
    options_json = nlohmann::json::parse(options_str);
  } catch (nlohmann::json::parse_error &e) {
    MS_LOG(EXCEPTION) << "Parse profiling option json failed, error:" << e.what();
  }
  auto iter = options_json.find(kOutputPath);
  if (iter != options_json.end() && iter->is_string()) {
    char real_path[PATH_MAX] = {0};
    if ((*iter).size() >= PATH_MAX) {
      MS_LOG(ERROR) << "Path is invalid for memory profiling.";
      return "";
    }
#if defined(_WIN32) || defined(_WIN64)
    if (_fullpath(real_path, common::SafeCStr(*iter), PATH_MAX) == nullptr) {
      MS_LOG(ERROR) << "Path is invalid for memory profiling.";
      return "";
    }
#else
    if (realpath(common::SafeCStr(*iter), real_path) == nullptr) {
      MS_LOG(ERROR) << "Path is invalid for memory profiling.";
      return "";
    }
#endif
    return real_path;
  }

  MS_LOG(ERROR) << "Output path is not found when save memory profiling data";
  return "";
}

void MemoryProfiling::SaveMemoryProfiling() {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  std::string dir_path = GetOutputPath();
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  std::string file = dir_path + std::string("/memory_usage_") + std::to_string(device_id) + std::string(".pb");

  MemoryToPB();

  std::fstream handle(file, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!memory_proto_.SerializeToOstream(&handle)) {
    MS_LOG(ERROR) << "Save memory profiling data to file failed";
  }
  handle.close();
  MS_LOG(INFO) << "Start save memory profiling data to " << file << " end";
  return;
}
}  // namespace profiler
}  // namespace mindspore
