/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "runtime/device/ascend/tasksink/task_generator.h"

#include <runtime/rt.h>
#include "backend/kernel_compiler/task_stream.h"
#include "utils/ms_utils.h"
#include "runtime/device/ascend/profiling/profiling_utils.h"
#include "runtime/device/ascend/profiling/profiling_manager.h"

namespace mindspore {
namespace device {
namespace ascend {
namespace tasksink {
bool TaskGenerator::GenTasks(const std::vector<CNodePtr> &anf_node_list, std::vector<TaskInfoPtr> *task_info_list,
                             uint32_t graph_id) {
  MS_LOG(INFO) << "GenTasks start...";
  MS_EXCEPTION_IF_NULL(task_info_list);
  // Traverse graph applykernel list and run
  if (!LaunchAllKernel(anf_node_list, task_info_list, graph_id)) {
    MS_LOG(ERROR) << "LaunchAllKernel failed";
    return false;
  }
  MS_LOG(INFO) << "GenTasks end...";
  return true;
}

void TaskGenerator::LaunchAddrCleanAkgKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  // akg process
  // set atomic clean addr
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, anf_node_ptr)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicOutputIndexs);
    auto graph = anf_node_ptr->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    auto manager = graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    auto node_users = manager->node_users();
    if (node_users[anf_node_ptr].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << anf_node_ptr->ToString() << " is empty.";
    }
    auto depend_node = node_users[anf_node_ptr].pop().first;
    if (!IsPrimitiveCNode(depend_node, prim::kPrimDepend)) {
      MS_LOG(EXCEPTION) << "Checking Depend node failed";
    }
    if (node_users[depend_node].empty()) {
      MS_LOG(EXCEPTION) << "Node users of " << depend_node->ToString() << " is empty.";
    }
    auto post_node = node_users[depend_node].pop().first;
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(post_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      input->size = device_address->size_;
      kernel_inputs->push_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size: " << clean_output_indexs.size();
  }
}

void TaskGenerator::LaunchAddrCleanKernel(const CNodePtr &anf_node_ptr, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  if (anf_node_ptr->inputs().size() != 2) {
    LaunchAddrCleanAkgKernel(anf_node_ptr, kernel_inputs);
    return;
  }
  MS_EXCEPTION_IF_NULL(anf_node_ptr->inputs()[1]);
  auto pre_node = (anf_node_ptr->inputs()[1])->cast<CNodePtr>();
  // set clean output addr
  if (AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
    auto clean_output_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
    for (auto index : clean_output_indexs) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      input->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(input->addr);
      input->size = device_address->size_;
      kernel_inputs->push_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexs.size();
  }
  // set clean workspace address
  if (AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
    auto clean_workspace_indexs = AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
    for (const auto &index : clean_workspace_indexs) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      MS_EXCEPTION_IF_NULL(workspace->addr);
      workspace->size = device_address->size_;
      kernel_inputs->push_back(workspace);
    }
  }
  auto clear_mems = AnfAlgo::GetNodeAttr<std::vector<size_t>>(anf_node_ptr, kAttrAtomicAddMemSize);
  if (kernel_inputs->size() != clear_mems.size()) {
    MS_LOG(EXCEPTION) << "AtomicAddClean kernel inputs size not equal clear memory size,kerenl_inputs size:"
                      << kernel_inputs->size() << ",clean mem size" << clear_mems.size();
  }
}

bool TaskGenerator::LaunchKernel(const CNodePtr &anf_node_ptr, uint32_t stream_id,
                                 std::vector<TaskInfoPtr> *task_info_list) {
  MS_EXCEPTION_IF_NULL(task_info_list);
  MS_EXCEPTION_IF_NULL(anf_node_ptr);
  AddressPtrList kernel_inputs;
  AddressPtrList kernel_workspaces;
  AddressPtrList kernel_outputs;
  auto kernel_mod = AnfAlgo::GetKernelMod(anf_node_ptr);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  kernel_mod->set_kernel_name(anf_node_ptr->fullname_with_scope());
  if (AnfAlgo::GetCNodeName(anf_node_ptr) != kAtomicAddrCleanOpName) {
    for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(anf_node_ptr); ++i) {
      auto real_input_index = AnfAlgo::GetRealInputIndex(anf_node_ptr, i);
      auto device_address = AnfAlgo::GetPrevNodeOutputAddr(anf_node_ptr, real_input_index);
      AddressPtr input = std::make_shared<Address>();
      input->addr = device_address->ptr_;
      input->size = device_address->size_;
      kernel_inputs.push_back(input);
    }

    for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(anf_node_ptr); ++i) {
      auto it = AnfAlgo::GetOutputAddr(anf_node_ptr, i);
      AddressPtr output = std::make_shared<Address>();
      output->addr = it->ptr_;
      output->size = it->size_;
      kernel_outputs.push_back(output);
    }

    for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(anf_node_ptr, i);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      workspace->addr = device_address->ptr_;
      workspace->size = device_address->size_;
      kernel_workspaces.push_back(workspace);
    }
  } else {
    LaunchAddrCleanKernel(anf_node_ptr, &kernel_inputs);
  }

  auto ascend_kernel_mod = dynamic_cast<kernel::AscendKernelMod *>(kernel_mod);
  MS_EXCEPTION_IF_NULL(ascend_kernel_mod);
  std::vector<TaskInfoPtr> task_info_ptrs =
    ascend_kernel_mod->GenTask(kernel_inputs, kernel_workspaces, kernel_outputs, stream_id);
  task_info_list->insert(task_info_list->end(), task_info_ptrs.begin(), task_info_ptrs.end());
  return true;
}

bool TaskGenerator::LaunchAllKernel(const std::vector<CNodePtr> &anf_node_list,
                                    std::vector<TaskInfoPtr> *task_info_list, uint32_t graph_id) {
  uint32_t current_op_index = 0;
  std::vector<CNodePtr> profiling_cnode_list;
  std::vector<std::string> kernel_name_list;
  for (const auto &anf_node_ptr : anf_node_list) {
    size_t old_size = task_info_list->size();
    uint32_t stream_id = AnfAlgo::GetStreamId(anf_node_ptr);
    MS_EXCEPTION_IF_NULL(anf_node_ptr);
    MS_LOG(INFO) << "Task gen launch begin, current_op_idx:" << current_op_index
                 << " name:" << anf_node_ptr->fullname_with_scope() << ", stream id:" << stream_id;
    if (!LaunchKernel(anf_node_ptr, stream_id, task_info_list)) {
      MS_LOG(ERROR) << "LaunchKernel failed.";
      return false;
    }
    for (size_t i = old_size; i < task_info_list->size(); ++i) {
      profiling_cnode_list.emplace_back(anf_node_ptr);
      kernel_name_list.emplace_back(anf_node_ptr->fullname_with_scope());
    }
    current_op_index++;
  }

  ProfilingUtils::SetGraphKernelName(graph_id, kernel_name_list);
  if (ProfilingManager::GetInstance().IsProfiling()) {
    ProfilingUtils::SetGraphProfilingCNode(graph_id, profiling_cnode_list);
  }
  return true;
}
}  // namespace tasksink
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
