/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "runtime/device/gsm/mem_usage_analyzer.h"
#include <memory>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
namespace mindspore {
namespace {
auto constexpr kInplaceNodeTypeSkip = "skip";
auto constexpr kInplaceNodeTypeAlgo = "inplace_algo";
}  // namespace
namespace device {
size_t MemUsageAnalyzer::AddTensorInfo(const AnfNodePtr &node, size_t index, bool is_workspace) {
  auto add_to_container = [this](const AnfNodePtr &node, size_t index,
                                 std::map<AnfNodePtr, std::map<size_t, size_t>> *container, bool is_workspace) {
    MS_EXCEPTION_IF_NULL(node);
    MS_EXCEPTION_IF_NULL(container);
    auto iter_node = container->find(node);
    if (iter_node != container->end()) {
      auto iter_tid = iter_node->second.find(index);
      if (iter_tid == iter_node->second.end()) {
        iter_node->second[index] = tensor_num_;
      } else {
        return iter_tid->second;
      }
    } else {
      (*container)[node] = std::map<size_t, size_t>({{index, tensor_num_}});
    }

    DeviceAddressPtr address = nullptr;
    if (is_workspace) {
      address = AnfAlgo::GetMutableWorkspaceAddr(node, index);
    } else {
      address = AnfAlgo::GetMutableOutputAddr(node, index, true);
    }

    MS_EXCEPTION_IF_NULL(address);
    auto info = std::make_shared<MemUsageTensorInfo>();
    info->tensor_id_ = tensor_num_;
    info->real_tensor_id_ = tensor_num_;
    info->tensor_size_ = address->GetSize();
    info->node_ = node;
    info->index_ = index;
    info->is_workspace_ = is_workspace;
    info->is_graph_input_ = !(node->isa<CNode>());
    info->is_graph_output_ = IsGraphOutput(node, index);
    (void)tensor_infos_.emplace_back(info);
    ++tensor_num_;
    return info->tensor_id_;
  };

  MS_EXCEPTION_IF_NULL(node);
  size_t tensor_id = 0;
  if (node->isa<ValueNode>()) {
    tensor_id = add_to_container(node, index, &kernel_input_value_tid_, false);
  } else if (node->isa<Parameter>()) {
    tensor_id = add_to_container(node, index, &kernel_input_param_tid_, false);
  } else if (is_workspace) {
    tensor_id = add_to_container(node, index, &kernel_workspace_tid_, true);
  } else {
    tensor_id = add_to_container(node, index, &kernel_output_tid_, false);
  }
  return tensor_id;
}

void MemUsageAnalyzer::Analyze(const KernelGraphPtr &graph) {
  AddOutputNodeInfo(graph);
  AddKernelAndTensorInfo(graph);
  AddFusedTensorInfo();
}

void MemUsageAnalyzer::AddOutputNodeInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    auto output_index = output_with_index.second;
    if (common::AnfAlgo::IsNopNode(output_node)) {
      auto real_node_with_index = common::AnfAlgo::GetPrevNodeOutput(output_node, output_index, true);
      output_node = real_node_with_index.first;
      output_index = real_node_with_index.second;
    }
    (void)graph_output_nodes_[output_node].insert(output_index);
  }
}

bool MemUsageAnalyzer::IsGraphOutput(const AnfNodePtr &node, size_t index) {
  auto iter = graph_output_nodes_.find(node);
  if (iter == graph_output_nodes_.end()) {
    return false;
  }

  if (iter->second.find(index) == iter->second.end()) {
    return false;
  }

  return true;
}

void MemUsageAnalyzer::AddFusedTensorInfo() {
  auto add_fused_tensor = [this](const std::vector<size_t> &tensors, size_t kernel_id) {
    if (tensors.size() <= 1) {
      return;
    }

    auto info = std::make_shared<MemUsageTensorInfo>();
    info->tensor_id_ = tensor_num_;
    info->real_tensor_id_ = tensor_num_;
    info->tensor_size_ = 0;
    info->node_ = nullptr;
    info->index_ = 0;
    (void)tensor_infos_.emplace_back(info);
    ++tensor_num_;

    for (auto tensor_id : tensors) {
      auto tensor_info = GetMemUsageTensorInfo(tensor_id);
      tensor_info->real_tensor_id_ = info->tensor_id_;
      info->tensor_size_ += tensor_info->tensor_size_;
      (void)info->fused_tensor_ids_.emplace_back(tensor_info->tensor_id_);
      (void)info->used_by_kernels_.emplace_back(kernel_id);
    }
  };

  for (size_t i = 0; i < kernel_infos_.size(); ++i) {
    auto &info = kernel_infos_[i];
    MS_EXCEPTION_IF_NULL(info);
    if (!info->is_comm_) {
      continue;
    }
    add_fused_tensor(info->input_tensors_, i);
    add_fused_tensor(info->output_tensors_, i);
  }
}

void MemUsageAnalyzer::AddKernelAndTensorInfo(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &exec_order = graph->execution_order();
  auto real_kernel_num = exec_order.size();
  kernel_infos_.resize(real_kernel_num);

  auto add_tensor_usage = [this](size_t tensor_id, size_t kernel_id, size_t *kernel_mem, bool inplace) {
    auto tensor_info = GetMemUsageTensorInfo(tensor_id);
    MS_EXCEPTION_IF_NULL(tensor_info);
    tensor_info->is_inplace_tensor_ = inplace;
    (void)tensor_info->used_by_kernels_.emplace_back(kernel_id);
    *kernel_mem += tensor_info->tensor_size_;
  };

  for (size_t i = 0; i < real_kernel_num; ++i) {
    const auto &node = exec_order[i];
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto kernel_info = std::make_shared<MemUsageKernelInfo>();
    kernel_info->is_comm_ = common::AnfAlgo::IsCommunicationOp(node);
    kernel_info->update_input_ = common::AnfAlgo::IsUpdateParameterKernel(node);
    bool inplace_node = common::AnfAlgo::IsInplaceNode(node, kInplaceNodeTypeSkip) ||
                        common::AnfAlgo::IsInplaceNode(node, kInplaceNodeTypeAlgo);

    // Memory used by this kernel
    size_t kernel_mem = 0;

    // Add input tensors
    const auto input_num = kernel_mod->GetInputSizeList().size();
    for (size_t index = 0; index < input_num; ++index) {
      auto prev_node_output = common::AnfAlgo::GetPrevNodeOutput(node, index, true);
      if (graph->IsInRefOutputMap(prev_node_output)) {
        prev_node_output = graph->GetRefCorrespondOutput(prev_node_output);
      }
      auto tensor_id = AddTensorInfo(prev_node_output.first, prev_node_output.second);
      (void)kernel_info->input_tensors_.emplace_back(tensor_id);
      add_tensor_usage(tensor_id, i, &kernel_mem, false);
    }

    // Add output tensors
    const auto output_num = kernel_mod->GetOutputSizeList().size();
    for (size_t index = 0; index < output_num; ++index) {
      if (graph->IsInRefOutputMap({node, index})) {
        auto real_node_pair = graph->GetRefCorrespondOutput({node, index});
        if (real_node_pair.first != node) {
          auto tensor_id = AddTensorInfo(real_node_pair.first, real_node_pair.second);
          (void)kernel_info->input_tensors_.emplace_back(tensor_id);
          add_tensor_usage(tensor_id, i, &kernel_mem, inplace_node);
        }
      } else {
        auto tensor_id = AddTensorInfo(node, index);
        (void)kernel_info->output_tensors_.emplace_back(tensor_id);
        add_tensor_usage(tensor_id, i, &kernel_mem, inplace_node);
      }
    }

    // Add workspace tensors
    const auto workspace_num = kernel_mod->GetWorkspaceSizeList().size();
    for (size_t index = 0; index < workspace_num; ++index) {
      auto tensor_id = AddTensorInfo(node, index, true);
      (void)kernel_info->workspace_tensors_.emplace_back(tensor_id);
      add_tensor_usage(tensor_id, i, &kernel_mem, false);
    }

    if (kernel_mem > least_mem_) {
      least_mem_ = kernel_mem;
    }

    kernel_infos_[i] = kernel_info;
  }
}
}  // namespace device
}  // namespace mindspore
