/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/hardware/ascend_somas.h"
#include <string>
#include <map>
#include <utility>
#include <vector>
#include "include/backend/optimizer/helper.h"
#include "utils/ms_context.h"
#include "plugin/device/ascend/hal/device/ascend_stream_assign.h"
#include "plugin/device/ascend/hal/profiler/memory_profiling.h"

namespace mindspore {
namespace device {
namespace ascend {
using TensorType = somas::TensorType;
using LifeLongType = somas::LifeLongType;
using mindspore::profiler::ascend::MemoryProfiling;
constexpr size_t ALONE = 1;

#ifndef ENABLE_SECURITY
void AscendSomas::ConvertToProfilingNode(uint32_t graph_id) const {
  if (!MemoryProfiling::GetInstance().IsMemoryProfilingInitialized()) {
    return;
  }
  auto graph_node = profiler::ascend::MemoryProfiling::GetInstance().GetGraphMemoryNode(graph_id);
  if (graph_node == nullptr) {
    graph_node = profiler::ascend::MemoryProfiling::GetInstance().AddGraphMemoryNode(graph_id);
    MS_LOG(INFO) << "Add graph memory node for dynamic memory profiling, graph id is " << graph_id;
  }

  for (const auto &tensor : tensors_list_) {
    profiler::ascend::TensorMemory tensor_memory;
    tensor_memory.SetTensorId(tensor->GetId());
    tensor_memory.SetAlignedSize(tensor->GetAlignedSize());
    tensor_memory.SetType(tensor->GetTypeString());
    tensor_memory.SetLifeStart(tensor->lifetime_.start_);
    tensor_memory.SetLifeEnd(tensor->lifetime_.end_);
    tensor_memory.SetLifeLong(tensor->GetLifelongString());
    graph_node->AddTensorMemory(tensor_memory);
  }

  for (const auto &node : nodes_list_) {
    profiler::ascend::NodeMemory node_memory;
    std::string name = GetSplitName(node->scope_full_name_);
    node_memory.SetNodeName(name);
    node_memory.SetNodeId(node->GetId());
    for (const auto &input_tensor : node->input_tensors_) {
      node_memory.AddInputTensorId(input_tensor->GetId());
    }
    for (const auto &output_tensor : node->output_tensors_) {
      node_memory.AddOutputTensorId(output_tensor->GetId());
    }
    for (const auto &workspace_tensor : node->workspace_tensors_) {
      node_memory.AddWorkSpaceTensorId(workspace_tensor->GetId());
    }
    graph_node->AddNodeMemory(node_memory);
  }
}
#endif

bool AscendSomas::Initialize() { return true; }

std::string AscendSomas::GetDeviceName() const { return "Ascend"; }

size_t AscendSomas::GetCommunicationReservedSize() const {
  constexpr size_t gap_size = 512;
  return gap_size;
}

size_t AscendSomas::GetAlignSize(size_t original_size) const {
  constexpr size_t alignment = 512;
  constexpr size_t alignment_complement = 31;
  size_t aligned_size =
    (original_size > 0) ? ((original_size + alignment + alignment_complement) / alignment) * alignment : 0;
  return aligned_size;
}

bool AscendSomas::GetDependExecOrderFlag(const session::KernelGraph &graph) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto task_sink = ms_context->get_param<bool>(MS_CTX_ENABLE_TASK_SINK);
  auto opt_level = ms_context->get_param<int>(MS_CTX_MEMORY_OPTIMIZE_LEVEL);
  if (task_sink || (opt_level != kOptimizeO0)) {
    return true;
  } else {
    return false;
  }
}

std::vector<vector<uint32_t>> AscendSomas::GetStreamGroupInfo() const {
  std::vector<vector<uint32_t>> stream_group;
  stream_group = device::ascend::AscendStreamAssign::GetInstance().get_stream_group();
  return stream_group;
}

std::map<std::string, UnReuseType> AscendSomas::GetUnReuseNodeType() const {
  std::map<std::string, UnReuseType> node_type;
  node_type[kGetNextOpName] = UnReuseType::kUnReuseOutput;
  return node_type;
}

bool AscendSomas::InitDevSpecControlTensors(const session::KernelGraph &graph) {
  InitEventInfo(graph);
  return true;
}

void AscendSomas::InitEventInfo(const session::KernelGraph &graph) {
  event_map_ = {};
  auto &kernels = graph.execution_order();
  for (const auto &kernel : kernels) {
    auto type = common::AnfAlgo::GetCNodeName(kernel);
    if (type == kSendOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel, kAttrEventId);
      auto iter = event_map_.find(event);
      if (iter == event_map_.end()) {
        auto pair = somas::EventPair();
        pair.send_ = kernel;
        event_map_[event] = pair;
      } else {
        iter->second.send_ = kernel;
      }
    } else if (type == kRecvOpName) {
      auto event = common::AnfAlgo::GetNodeAttr<uint32_t>(kernel, kAttrEventId);
      auto iter = event_map_.find(event);
      if (iter == event_map_.end()) {
        auto pair = somas::EventPair();
        pair.recv_ = kernel;
        event_map_[event] = pair;
      } else {
        iter->second.recv_ = kernel;
      }
    }
  }

  for (auto &event : event_map_) {
    auto send_iter = nodes_map_.find(event.second.send_.get());
    auto recv_iter = nodes_map_.find(event.second.recv_.get());
    if (send_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find Ascend somas node for " << event.second.send_->fullname_with_scope();
      continue;
    }
    if (recv_iter == nodes_map_.end()) {
      MS_LOG(WARNING) << "Can't find Ascend somas node for " << event.second.recv_->fullname_with_scope();
      continue;
    }
    AddControlTensor(send_iter->second.at(0), recv_iter->second.at(0));
  }
  MS_LOG(DEBUG) << "Ascend Somas InitEventInfo end.";
}

bool AscendSomas::DevSpecNodeProcess(const session::KernelGraph &graph) {
  IndependentNodeOutputProcess(graph);
  NonTaskSplitProcess(graph);
  return true;
}

void AscendSomas::IndependentNodeOutputProcess(const session::KernelGraph &graph) {
  auto &kernel_cnodes = graph.execution_order();
  size_t total_size = 0;
  for (const auto &kernel : kernel_cnodes) {
    bool independent = AnfAlgo::IsIndependentNode(kernel);
    if (!independent) {
      continue;
    }
    auto iter = nodes_map_.find(kernel.get());
    if (iter != nodes_map_.end()) {
      auto &node = iter->second.at(0);
      MS_EXCEPTION_IF_NULL(node);
      auto semi_reuse_output_tensors = node->output_tensors_;
      for (auto &tensor : semi_reuse_output_tensors) {
        MS_EXCEPTION_IF_NULL(tensor);
        total_size += tensor->GetAlignedSize();
        tensor->lifelong_value_ = LifeLongType::kLifeLongGraphEnd;
      }
    }
  }

  MS_LOG(INFO) << "Special Tensor total size: Independent Node output " << total_size;
}

void AscendSomas::NonTaskSplitProcess(const session::KernelGraph &graph) {
  // When not used task sink mode, should not process non task split mem-reuse.
  // Because the logic for processing non task split memory offset is in the function TaskGenerator:: GetTaskInput,
  // only run in task sink mode.
  if (!graph.is_graph_run_mode()) {
    return;
  }
  auto &kernel_cnodes = graph.execution_order();
  for (const auto &kernel : kernel_cnodes) {
    auto op_name = common::AnfAlgo::GetCNodeName(kernel);
    if (common::AnfAlgo::IsNonTaskOp(kernel)) {
      std::vector<size_t> refnode_input_output;
      auto node = nodes_map_[kernel.get()].at(0);
      MS_EXCEPTION_IF_NULL(node);
      if (node->input_tensors_.empty()) {
        MS_LOG(EXCEPTION) << op_name << " has no input tensor, can not do split non_task process.";
      }
      auto input_tensor = node->input_tensors_[0];
      MS_EXCEPTION_IF_NULL(input_tensor);
      input_tensor->type_ = TensorType::kUnion;
      refnode_input_output.push_back(input_tensor->GetId());

      for (auto &output_tensor : node->output_tensors_) {
        MS_EXCEPTION_IF_NULL(output_tensor);
        output_tensor->type_ = TensorType::kUnion;
        refnode_input_output.push_back(output_tensor->GetId());
      }
      union_tensors_list_.push_back(refnode_input_output);
    }
  }
}

void AscendSomas::CommunicationTensorProcess(const std::vector<somas::SomasTensorPtr> &tensors) const {
  // add gap for first and last input
  if (tensors.size() != ALONE) {
    for (auto &tensor : tensors) {
      MS_EXCEPTION_IF_CHECK_FAIL(tensor->aligned_size_ != 0, "The size of communication tensor is zero, tensor id: " +
                                                               std::to_string(tensor->GetId()));
    }
  }
  if (tensors[0]->aligned_size_ != 0) {
    tensors[0]->aligned_size_ += GetCommunicationReservedSize();
  }
  if (tensors[tensors.size() - 1]->aligned_size_ != 0) {
    tensors[tensors.size() - 1]->aligned_size_ += GetCommunicationReservedSize();
  }
}

bool AscendSomas::NeedContiguous(const std::vector<size_t> &inputs) const { return true; }
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
