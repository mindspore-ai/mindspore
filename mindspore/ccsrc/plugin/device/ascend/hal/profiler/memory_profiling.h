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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PROFILING_MEMORY_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PROFILING_MEMORY_H

#include "proto/memory_profiling.pb.h"
#include <string>
#include <map>
#include <vector>
#include <memory>
#include "utils/ms_context.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace profiler {
namespace ascend {
class NodeMemory {
 public:
  NodeMemory() : node_name_(""), node_id_(0) {}
  ~NodeMemory() = default;

  void SetNodeName(const std::string &name) { node_name_ = name; }
  void SetNodeId(uint64_t node_id) { node_id_ = node_id; }
  void AddInputTensorId(uint64_t node_id) { input_tensor_id_.emplace_back(node_id); }
  void AddOutputTensorId(uint64_t node_id) { output_tensor_id_.emplace_back(node_id); }
  void AddWorkSpaceTensorId(uint64_t node_id) { workspace_tensor_id_.emplace_back(node_id); }
  std::string GetNodeName() const { return node_name_; }
  uint64_t GetNodeId() const { return node_id_; }
  std::vector<uint64_t> GetInputTensorId() const { return input_tensor_id_; }
  std::vector<uint64_t> GetOutputTensorId() const { return output_tensor_id_; }
  std::vector<uint64_t> GetWorkspaceTensorId() const { return workspace_tensor_id_; }

 private:
  std::string node_name_;
  uint64_t node_id_;
  std::vector<uint64_t> input_tensor_id_;
  std::vector<uint64_t> output_tensor_id_;
  std::vector<uint64_t> workspace_tensor_id_;
};

class TensorMemory {
 public:
  TensorMemory() : tensor_id_(0), size_(0), type_(""), life_start_(0), life_end_(0), life_long_("") {}
  ~TensorMemory() = default;

  void SetTensorId(uint64_t tensor_id) { tensor_id_ = tensor_id; }
  void SetAlignedSize(uint64_t size) { size_ = size; }
  void SetType(const std::string &type) { type_ = type; }
  void SetLifeStart(uint64_t start) { life_start_ = start; }
  void SetLifeEnd(uint64_t end) { life_end_ = end; }
  void SetLifeLong(const std::string &life_long) { life_long_ = life_long; }
  uint64_t GetTensorId() const { return tensor_id_; }
  uint64_t GetAlignedSize() const { return size_; }
  std::string GetType() const { return type_; }
  uint64_t GetLifeStart() const { return life_start_; }
  uint64_t GetLifeEnd() const { return life_end_; }
  std::string GetLifeLong() const { return life_long_; }

 private:
  uint64_t tensor_id_;
  uint64_t size_;          // aligned tensor size
  std::string type_;       // see TensorType in somas_tensor.h
  uint64_t life_start_;    // the exe node id at which tensor memory allocated
  uint64_t life_end_;      // the exe node id at which tensor memory deallocated
  std::string life_long_;  // see LifeLongType in somas_tensor.h
};

class GraphMemory {
 public:
  explicit GraphMemory(uint32_t graph_id) : graph_id_(graph_id), static_mem_size_(0) {}
  ~GraphMemory() = default;
  void AddStaticMemorySize(uint32_t size) { static_mem_size_ += size; }
  void AddNodeMemory(const NodeMemory &node) { node_memory_.emplace_back(node); }
  void AddTensorMemory(const TensorMemory &node) { tensor_memory_.emplace_back(node); }
  uint32_t GetGraphId() const { return graph_id_; }
  uint32_t GetStaticMemSize() const { return static_mem_size_; }
  std::vector<NodeMemory> GetNodeMemory() const { return node_memory_; }
  std::vector<TensorMemory> GetTensorMemory() const { return tensor_memory_; }

 private:
  uint32_t graph_id_;
  uint32_t static_mem_size_;
  std::vector<NodeMemory> node_memory_;
  std::vector<TensorMemory> tensor_memory_;
};

class MemoryProfiling {
 public:
  MemoryProfiling() : device_mem_size_(0), is_initialized_(false), is_enabled_(false), has_save_memory_data_(false) {}
  ~MemoryProfiling() = default;

  static MemoryProfiling &GetInstance() {
    static MemoryProfiling instance;
    return instance;
  }

  std::shared_ptr<GraphMemory> AddGraphMemoryNode(uint32_t graph_id);
  std::shared_ptr<GraphMemory> GetGraphMemoryNode(uint32_t graph_id) const;
  void SetDeviceMemSize(uint64_t size) { device_mem_size_ = size; }
  bool MemoryToPB();
  void SaveMemoryProfiling();
  bool IsMemoryProfilingInitialized() const { return is_initialized_; }
  bool IsMemoryProfilingEnabled() const { return is_enabled_; }
  void SetMemoryProfilingInitialize(const std::string &profiling_options);
  bool NeedSaveMemoryProfiling() { return (is_enabled_) && (graph_memory_.size() != 0) && (!has_save_memory_data_); }
  void StartMemoryProfiling();
  void StopMemoryProfiling();

 private:
  MemoryProto memory_proto_;
  std::map<uint32_t, std::shared_ptr<GraphMemory>> graph_memory_;
  uint64_t device_mem_size_;
  bool is_initialized_;
  bool is_enabled_;
  bool has_save_memory_data_;
};
}  // namespace ascend
}  // namespace profiler
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_HAL_PROFILER_PROFILING_MEMORY_H
