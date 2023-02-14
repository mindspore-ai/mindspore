/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "runtime/device/kernel_runtime.h"
#include <algorithm>
#include <functional>
#include <utility>
#include <vector>
#include <set>
#include <shared_mutex>
#include "backend/common/optimizer/helper.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "backend/common/optimizer/dynamic_shape/dynamic_shape_helper.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/kernel_graph.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "runtime/pynative/op_runtime_info.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "debug/data_dump/dump_json_parser.h"
#include "frontend/operator/ops.h"
#include "ir/value.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "utils/shape_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "include/common/debug/env_config_parser.h"
#include "kernel/common_utils.h"

using mindspore::kernel::Address;
using mindspore::kernel::AddressPtr;

namespace mindspore {
namespace device {
constexpr size_t kAtomicCleanInputSize = 2;
namespace {
std::vector<AnfNodePtr> GetGraphInputs(const session::KernelGraph &graph) {
  auto graph_inputs = graph.inputs();
  std::vector<AnfNodePtr> result(graph_inputs.begin(), graph_inputs.end());
  std::set<AnfNodePtr> inputs_set(graph_inputs.begin(), graph_inputs.end());
  auto kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t i = 0; i < input_num; ++i) {
      auto input_node = kernel->input(i + 1);
      auto input_real_node = common::AnfAlgo::VisitKernelWithReturnType(input_node, 0).first;
      MS_EXCEPTION_IF_NULL(input_real_node);
      if (input_real_node->isa<Parameter>() && inputs_set.find(input_real_node) == inputs_set.end()) {
        (void)inputs_set.insert(input_real_node);
        (void)result.emplace_back(input_real_node);
      }
    }
  }
  return result;
}

// Check whether mutex exists for a stream.
std::pair<bool, std::mutex *> CheckStreamMutexExist(
  const void *stream, const mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> &mtxs_for_streams,
  std::shared_mutex *shd_mtx) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  std::shared_lock<std::shared_mutex> shd_lock(*shd_mtx);
  auto iter = mtxs_for_streams.find(stream);
  if (iter != mtxs_for_streams.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    return std::make_pair(true, iter->second.get());
  }
  return std::make_pair(false, nullptr);
}

// Create a mutex for stream.
std::mutex *CreateStreamMutex(const void *stream, std::shared_mutex *shd_mtx,
                              mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> *mtxs_for_streams) {
  MS_EXCEPTION_IF_NULL(stream);
  MS_EXCEPTION_IF_NULL(shd_mtx);
  MS_EXCEPTION_IF_NULL(mtxs_for_streams);

  std::unique_lock<std::shared_mutex> unq_lock(*shd_mtx);
  auto ret_pair = mtxs_for_streams->emplace(stream, std::make_shared<std::mutex>());

  MS_EXCEPTION_IF_NULL(ret_pair.first->second);
  return ret_pair.first->second.get();
}

bool IsNeedAllocMem(const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &graph = node->func_graph();
  if (graph == nullptr) {
    return true;
  }
  if (!graph->has_flag(kFlagEnableZeroCopyInGraph)) {
    return true;
  }
  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  return std::find_if(outputs.begin(), outputs.end(), [&node, &index](const KernelWithIndex &output) {
           const auto &real_output = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
           return ((real_output.first == node) && (real_output.second == index));
         }) == outputs.end();
}
}  // namespace
constexpr size_t kMinInputSize = 2;
KernelRuntime::TbeLaunchKernelModCallBack KernelRuntime::tbe_call_ = nullptr;
KernelRuntime::~KernelRuntime() {
  stream_ = nullptr;
  communication_stream_ = nullptr;
}

std::lock_guard<std::mutex> KernelRuntime::LockRuntime(const void *stream) {
  MS_EXCEPTION_IF_NULL(stream);
  // Read-write lock for accessing mtxs_for_streams map.
  // When the lock of each stream is created, mtxs_for_streams can be accessed concurrently to improve performance.
  static std::shared_mutex shd_mtx;
  static mindspore::HashMap<const void *, std::shared_ptr<std::mutex>> mtxs_for_streams;

  std::mutex *stream_mtx = nullptr;
  // Check whether mutex exists for a stream.
  std::pair<bool, std::mutex *> ret_pair = CheckStreamMutexExist(stream, mtxs_for_streams, &shd_mtx);
  if (ret_pair.first) {
    stream_mtx = ret_pair.second;
  } else {
    // Create a mutex for stream.
    stream_mtx = CreateStreamMutex(stream, &shd_mtx, &mtxs_for_streams);
  }

  MS_EXCEPTION_IF_NULL(stream_mtx);
  return std::lock_guard<std::mutex>(*stream_mtx);
}

bool KernelRuntime::Load(const session::KernelGraph &, bool) {
  MS_LOG(INFO) << "Call default load.";
  return true;
}

bool KernelRuntime::LoadData(const session::KernelGraph &) {
  MS_LOG(INFO) << "Call default load data.";
  return false;
}

bool KernelRuntime::NodeOutputDeviceAddressExist(const AnfNodePtr &kernel, size_t index) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (AnfAlgo::OutputAddrExist(kernel, index)) {
    // In subgraph sink mode, if the kernel does not need allocate memory, it cannot be skipped.
    const auto &address = AnfAlgo::GetOutputAddr(kernel, index, IsNeedAllocMem(kernel, index));
    MS_EXCEPTION_IF_NULL(address);
    return address->GetDeviceType() == GetTargetDeviceType();
  }
  return false;
}

void KernelRuntime::AssignMemory(const session::KernelGraph &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (UseMemScheduler()) {
    AssignStaticMemoryValueNode(graph);
    ResetNodeAddress(graph);
    AddCommunicationMemInfo(graph);
  } else {
    MS_EXCEPTION_IF_NULL(mem_manager_);
    mem_manager_->ResetDynamicMemory();
    AssignStaticMemory(graph);
    AssignDynamicMemory(graph);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::GetCommunicationInputInfo(const AnfNodePtr &node, size_t *total_size,
                                              DeviceAddressPtrList *address_list,
                                              std::vector<size_t> *align_size_list) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(total_size);
  MS_EXCEPTION_IF_NULL(address_list);
  MS_EXCEPTION_IF_NULL(align_size_list);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i, true);
    auto input_node = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    DeviceAddressPtr address = nullptr;
    if (AnfAlgo::OutputAddrExist(input_node, input_node_with_index.second)) {
      address = AnfAlgo::GetMutableOutputAddr(input_node, input_node_with_index.second);
    } else {
      address = PreAssignCNodeMemory(input_node, input_node_with_index.second);
    }
    MS_EXCEPTION_IF_NULL(address);
    auto align_size = MemoryManager::GetCommonAlignSize(address->size());
    *total_size += align_size;
    (void)address_list->emplace_back(address);
    (void)align_size_list->emplace_back(align_size);
  }
}

void KernelRuntime::AssignCommunicationInputFromMemoryPool(const AnfNodePtr &node) const {
  if (!common::AnfAlgo::IsCommunicationOp(node)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  size_t total_size = 0;
  DeviceAddressPtrList address_list;
  std::vector<size_t> align_size_list;
  GetCommunicationInputInfo(node, &total_size, &address_list, &align_size_list);
  if (align_size_list.empty()) {
    MS_LOG(WARNING) << "No inputs for " << node->fullname_with_scope();
    return;
  }

  if (!mem_manager_->MallocContinuousMemFromMemPool(address_list, total_size, align_size_list)) {
    MS_LOG(EXCEPTION) << "Allocate continuous memory failed, totol_size:" << total_size;
  }
}

void KernelRuntime::GetCommunicationOutputInfo(const AnfNodePtr &node, size_t *total_size,
                                               DeviceAddressPtrList *address_list,
                                               std::vector<size_t> *align_size_list) const {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(total_size);
  MS_EXCEPTION_IF_NULL(align_size_list);
  MS_EXCEPTION_IF_NULL(address_list);

  const auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  const auto output_size_list = kernel_mod->GetOutputSizeList();
  for (size_t i = 0; i < output_size_list.size(); ++i) {
    DeviceAddressPtr address = nullptr;
    if (AnfAlgo::OutputAddrExist(node, i)) {
      address = AnfAlgo::GetMutableOutputAddr(node, i);
    } else {
      const std::string output_format = AnfAlgo::GetOutputFormat(node, i);
      const auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
      const auto tensor_size = AnfAlgo::GetOutputTensorMemSize(node, i);
      address = CreateDeviceAddress(nullptr, tensor_size, output_format, output_type, {node, i});
      AnfAlgo::SetOutputAddr(address, i, node.get());
    }
    MS_EXCEPTION_IF_NULL(address);
    auto align_size = MemoryManager::GetCommonAlignSize(address->size());
    *total_size += align_size;
    align_size_list->emplace_back(align_size);
    (void)address_list->emplace_back(address);
  }
}

void KernelRuntime::AssignCommunicationOutputFromMemoryPool(const AnfNodePtr &node) const {
  if (!common::AnfAlgo::IsCommunicationOp(node)) {
    return;
  }
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  size_t total_size = 0;
  std::vector<size_t> align_size_list;
  std::vector<DeviceAddressPtr> address_list;
  GetCommunicationOutputInfo(node, &total_size, &address_list, &align_size_list);
  if (align_size_list.empty()) {
    MS_LOG(WARNING) << "No output for " << node->fullname_with_scope();
    return;
  }

  if (!mem_manager_->MallocContinuousMemFromMemPool(address_list, total_size, align_size_list)) {
    MS_LOG(EXCEPTION) << "Allocate continuous memory failed, totol_size:" << total_size;
  }
}

void KernelRuntime::RunOpMallocPre(const session::KernelGraph &graph,
                                   const std::vector<tensor::TensorPtr> &input_tensors) {
  const auto &nodes = graph.execution_order();
  // Malloc for Node output
  for (const auto &node : nodes) {
    auto output_num = AnfAlgo::GetOutputTensorNum(node);
    for (size_t i = 0; i < output_num; ++i) {
      MS_EXCEPTION_IF_NULL(node);
      auto runtime_info = node->user_data<runtime::OpRuntimeInfo>();
      MS_EXCEPTION_IF_NULL(runtime_info);
      auto const &output_format = runtime_info->output_format(i);
      auto output_type = runtime_info->output_type(i);
      auto tensor_size = runtime_info->output_tensor_size(i);
      // Create DeviceAddress without ptr.
      // Get real device ptr after KernelBuild finish.
      auto device_address = CreateDeviceAddress(nullptr, tensor_size, output_format, output_type);
      device_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
      AnfAlgo::SetOutputAddr(device_address, i, node.get());
    }
  }

  // Malloc for graph input
  if (input_tensors.size() != graph.inputs().size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size()
                      << " should be equal to graph input parameter size " << graph.inputs().size();
  }
  for (size_t input_index = 0; input_index < graph.inputs().size(); ++input_index) {
    auto item = graph.inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      auto current_tensor = input_tensors[input_index];
      MS_EXCEPTION_IF_NULL(current_tensor);
      auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(current_tensor->device_address());
      if (output_address != nullptr && output_address->GetDeviceType() == GetTargetDeviceType()) {
        AnfAlgo::SetOutputAddr(output_address, index, item.get());
        continue;
      }
      auto op_runtime_info = item->user_data<runtime::OpRuntimeInfo>();
      MS_EXCEPTION_IF_NULL(op_runtime_info);
      TypeId output_type_id = op_runtime_info->output_type(index);
      auto output_tensor_size = op_runtime_info->output_tensor_size(index);
      auto output_format = op_runtime_info->output_format(index);
      auto device_address =
        CreateDeviceAddress(nullptr, output_tensor_size, output_format, output_type_id, {item, index});
      device_address->set_from_persistent_mem(current_tensor->is_parameter());
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
      current_tensor->set_device_address(device_address);
      current_tensor->set_sync_status(kNeedSyncHostToDevice);
    }
  }
}

void KernelRuntime::ResetNodeAddress(const session::KernelGraph &kernel_graph) {
  auto kernels = kernel_graph.execution_order();
  for (auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_num; ++j) {
      auto input_index = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, j);
      KernelWithIndex kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, input_index, true);
      auto index = kernel_with_index.second;
      auto &input_node = kernel_with_index.first;
      if (NodeOutputDeviceAddressExist(input_node, index)) {
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(input_node, index);
      if (output_type_id == kTypeUnknown) {
        MS_LOG(WARNING) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(input_node, index);
      auto device_address = CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(input_node, index),
                                                output_type_id, {input_node, index});
      AnfAlgo::SetOutputAddr(device_address, index, input_node.get());
    }

    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      auto output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      AnfAlgo::SetOutputAddr(CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type), i,
                             kernel.get());
    }
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(nullptr, workspace_sizes[i], kOpFormat_DEFAULT, kNumberTypeFloat32),
                                i, kernel.get());
    }
  }
}

void KernelRuntime::RunOpAssignMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                      const session::KernelGraph &graph, bool is_gradient_out,
                                      const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();

  for (const auto &node : graph.execution_order()) {
    AssignCommunicationOutputFromMemoryPool(node);
    AssignCommunicationInputFromMemoryPool(node);
  }

  RunOpAssignInputMemory(input_tensors, graph);
  AssignStaticMemoryValueNode(graph);
  for (const auto &node : graph.execution_order()) {
    RunOpAssignOutputMemory(node, tensor_to_node, is_gradient_out);
    RunOpAssignWorkSpaceMemory(node);
  }
  UpdateRefNodeOutputMem(graph);
}

void KernelRuntime::RunOpClearMemory(const session::KernelGraph &graph) const {
  // clear input parameter memory resource
  for (const auto &input_node : graph.inputs()) {
    MS_EXCEPTION_IF_NULL(input_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, input_node.get());
  }
  // clear input value node memory resource
  for (const auto &value_node : graph.graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(value_node);
    AnfAlgo::SetOutputAddr(nullptr, 0, value_node.get());
  }
  for (const auto &cnode : graph.execution_order()) {
    MS_EXCEPTION_IF_NULL(cnode);
    // clear output memory resource
    size_t output_num = AnfAlgo::GetOutputTensorNum(cnode);
    for (size_t index = 0; index < output_num; ++index) {
      AnfAlgo::SetOutputAddr(nullptr, index, cnode.get());
    }
    // clear workspace memory resource
    auto kernel_mod = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t index = 0; index < workspace_lists.size(); ++index) {
      AnfAlgo::SetWorkspaceAddr(nullptr, index, cnode.get());
    }
  }
}

#ifdef ENABLE_DEBUGGER
bool KernelRuntime::DumpDataEnabled() {
  // Returns true if e2e dump is enabled.
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  return dump_json_parser.e2e_dump_enabled();
}

bool KernelRuntime::DumpDataEnabledIteration() {
  // Returns true if e2e dump is enabled and current iteration must be dumped.
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (!dump_json_parser.e2e_dump_enabled()) {
    return false;
  }

  auto cur_iter = dump_json_parser.cur_dump_iter();
  if (dump_json_parser.IsDumpIter(cur_iter)) {
    return true;
  }
  return false;
}
#endif

void KernelRuntime::AssignStaticMemory(const session::KernelGraph &graph) {
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  AssignStaticMemoryOutput(graph);
}

void KernelRuntime::RunOpAssignInputMemory(const std::vector<tensor::TensorPtr> &input_tensors,
                                           const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (input_tensors.size() != graph.inputs().size()) {
    MS_LOG(EXCEPTION) << "Input tensors size " << input_tensors.size()
                      << " should be equal to graph input parameter size " << graph.inputs().size();
  }

  for (size_t input_index = 0; input_index < graph.inputs().size(); ++input_index) {
    auto item = graph.inputs()[input_index];
    MS_EXCEPTION_IF_NULL(item);
    if (!item->isa<Parameter>()) {
      continue;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      auto current_tensor = input_tensors[input_index];
      MS_EXCEPTION_IF_NULL(current_tensor);
      auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(current_tensor->device_address());
      // Device address have already create
      if (output_address != nullptr && output_address->GetDeviceType() == GetTargetDeviceType()) {
        if (output_address->ptr_ == nullptr) {
          if (!mem_manager_->MallocMemFromMemPool(output_address, output_address->size())) {
            MS_LOG(EXCEPTION) << "Allocate memory failed, size:" << output_address->size();
          }
        }

        AnfAlgo::SetOutputAddr(output_address, index, item.get());
        continue;
      }
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      if (output_type_id == kTypeUnknown) {
        output_type_id = common::AnfAlgo::GetOutputInferDataType(item, index);
      }
      auto tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
      // Device address new create
      auto device_address =
        CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id, {item, index});
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(mem_manager_);
      device_address->set_from_persistent_mem(true);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, tensor_size);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << tensor_size;
      }
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputMemory(const AnfNodePtr &kernel,
                                            const std::map<tensor::TensorPtr, session::KernelWithIndex> &tensor_to_node,
                                            bool is_gradient_out) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }

  // Use device_address Allocated in RunOpMallocPre.
  for (auto &iter : tensor_to_node) {
    auto device_address = iter.first->device_address();
    AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(device_address), iter.second.second,
                           iter.second.first.get());
  }

  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if (AnfAlgo::OutputAddrExist(kernel, i, false)) {
      auto address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
      MS_EXCEPTION_IF_NULL(address);
      if (address->ptr() == nullptr) {
        MS_EXCEPTION_IF_NULL(mem_manager_);
        if (!mem_manager_->MallocMemFromMemPool(address, address->size())) {
          MS_LOG(EXCEPTION) << "Allocate memory failed, size:" << address->size();
        }
      }
      continue;
    }
    if (common::AnfAlgo::GetCNodeName(kernel) == kApplyMomentumOpName ||
        common::AnfAlgo::GetCNodeName(kernel) == kApplyMomentumDOpName) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
      continue;
    }
    std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type, {kernel, i});
    MS_EXCEPTION_IF_NULL(device_address);
    device_address->set_host_shape(trans::GetRuntimePaddingShape(kernel, i));
    if (is_gradient_out) {
      device_address->set_from_persistent_mem(true);
    }
    auto ret = mem_manager_->MallocMemFromMemPool(device_address, output_sizes[i]);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << output_sizes[i];
    }
    AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
  }
}

void KernelRuntime::RunOpAssignWorkSpaceMemory(const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (kernel->isa<CNode>()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_lists = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_lists.size(); ++i) {
      auto device_address = CreateDeviceAddress(nullptr, workspace_lists[i], "", kTypeUnknown);
      MS_EXCEPTION_IF_NULL(device_address);
      auto ret = mem_manager_->MallocMemFromMemPool(device_address, workspace_lists[i]);
      if (!ret) {
        MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << workspace_lists[i];
      }
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void KernelRuntime::RunOpAssignOutputNodeMemory(const ValuePtr &pre_output_value,
                                                const session::KernelGraph &graph) const {
  if (pre_output_value == nullptr) {
    return;
  }
  std::vector<tensor::TensorPtr> pre_output_tensors;
  TensorValueToTensor(pre_output_value, &pre_output_tensors);
  auto output_nodes = graph.outputs();
  if (pre_output_tensors.size() != output_nodes.size()) {
    MS_LOG(EXCEPTION) << "The size of pre output tensors [" << pre_output_tensors.size()
                      << "] is not equal to the size of output nodes of graph [" << output_nodes.size() << "]";
  }
  // share output address with pre output tensors
  for (size_t i = 0; i < output_nodes.size(); ++i) {
    auto output_node_with_index = common::AnfAlgo::VisitKernel(output_nodes[i], 0);
    auto output_node = output_node_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    if (!output_node->isa<CNode>()) {
      if (output_node->isa<Parameter>()) {
        auto param = output_node->cast<ParameterPtr>();
        if (param != nullptr && !param->has_default()) {
          MS_LOG(EXCEPTION) << "The output parameter should be real parameter!";
        }
      }
      continue;
    }
    auto real_output_cnode = output_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(real_output_cnode);
    MS_EXCEPTION_IF_NULL(pre_output_tensors[i]);
    if (pre_output_tensors[i]->device_address() == nullptr) {
      MS_LOG(INFO) << "The address of pre output tensor [" << i << "] is a nullptr!";
      continue;
    }
    if (common::AnfAlgo::IsNopNode(real_output_cnode)) {
      if (real_output_cnode->inputs().size() < kMinInputSize) {
        MS_LOG(EXCEPTION) << "The input size of output node: " << real_output_cnode->DebugString()
                          << " should large than one!";
      }
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, real_output_cnode->input(1).get());
    } else {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(pre_output_tensors[i]->device_address()),
                             output_node_with_index.second, output_node_with_index.first.get());
    }
  }
}

void KernelRuntime::AssignStaticMemoryInput(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (graph.need_inline()) {
    return;
  }
  MS_LOG(INFO) << "AssignStaticMemoryInput start for graph " << graph.graph_id();
  auto graph_inputs = GetGraphInputs(graph);
  auto graph_valid_input = graph.valid_inputs();
  graph_inputs.insert(graph_inputs.end(), graph.child_graph_result().begin(), graph.child_graph_result().end());
  std::vector<AnfNodePtr> need_alloc_nodes;
  auto add_need_alloc_nodes = [&need_alloc_nodes, this](const AnfNodePtr &node) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      return;
    }
    if (NodeOutputDeviceAddressExist(node, 0)) {
      const auto &address = AnfAlgo::GetOutputAddr(node, 0);
      MS_EXCEPTION_IF_NULL(address);
      if (address->GetPtr() != nullptr) {
        return;
      }
    }
    need_alloc_nodes.push_back(node);
  };

  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto input_node = graph_inputs[i];
    MS_EXCEPTION_IF_NULL(input_node);
    if (i < graph_valid_input.size() && !graph_valid_input[i]) {
      continue;
    }
    if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimMakeTuple)) {
      auto outs = common::AnfAlgo::GetAllOutput(input_node);
      for (auto &out : outs) {
        MS_EXCEPTION_IF_NULL(out);
        add_need_alloc_nodes(out);
      }
    }
    add_need_alloc_nodes(input_node);
  }
  std::map<AnfNodePtr, AnfNodePtr> shadow_backend_node_map;
  GetShadowBackendNodeMap(graph, &shadow_backend_node_map);
  for (auto &item : need_alloc_nodes) {
    MS_EXCEPTION_IF_NULL(item);
    auto output_size = AnfAlgo::GetOutputTensorNum(item);
    for (size_t index = 0; index < output_size; index++) {
      TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
      // if graph output is a weight and doesn't link to any cnode, it's data type will be unknown
      if (output_type_id == kTypeUnknown) {
        MS_LOG(INFO) << "It is not suggested to use a lonely weight parameter as the output of graph";
        continue;
      }
      DeviceAddressPtr device_address = GetInternalDeviceAddress(graph, item);
      GetDeviceAddress(item, shadow_backend_node_map, index, graph, &device_address);
      AnfAlgo::SetOutputAddr(device_address, index, item.get());
    }
  }
  MS_LOG(INFO) << "AssignStaticMemoryInput end";
}

void KernelRuntime::GetDeviceAddress(const AnfNodePtr &item,
                                     const std::map<AnfNodePtr, AnfNodePtr> shadow_backend_node_map, size_t index,
                                     const session::KernelGraph &graph, DeviceAddressPtr *device_address) {
  AnfNodePtr shadow_node = nullptr;
  auto iter = shadow_backend_node_map.find(item);
  if (iter != shadow_backend_node_map.end()) {
    shadow_node = iter->second;
  }
  if (*device_address == nullptr && shadow_node != nullptr) {
    auto conj_device_address = AnfAlgo::GetMutableOutputAddr(shadow_node, index);
    if (conj_device_address != nullptr && conj_device_address->GetDeviceType() == DeviceType::kAscend) {
      *device_address = conj_device_address;
    }
  } else if (*device_address == nullptr) {
    auto tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(item, index);
    *device_address =
      CreateDeviceAddress(nullptr, tensor_size, AnfAlgo::GetOutputFormat(item, index), output_type_id, {item, index});
  }

  // Set the flag of no user parameter and not malloc memory.
  if ((*device_address != nullptr) && item->isa<Parameter>()) {
    auto input_param = item->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(input_param);
    // Unused address will not alloc memory, which is easy to cause problems for weight node, so skip weight node.
    if (!common::AnfAlgo::IsParameterWeight(input_param) && !input_param->IsUsedByRealKernelInGraph(graph.graph_id())) {
      MS_LOG(INFO) << "Node:" << item->fullname_with_scope() << " debug name:" << item->DebugString()
                   << " is not used in the graph " << graph.graph_id();
      (*device_address)->UpdateFlag(kDeviceAddressFlagNotUsed);
      return;
    }
  }

  if (*device_address != nullptr && (*device_address)->GetPtr() == nullptr) {
    auto tensor_size = AnfAlgo::GetOutputTensorMemSize(item, index);
    (*device_address)->set_host_shape(trans::GetRuntimePaddingShape(item, index));
    MS_LOG(INFO) << "Assign Static Memory for Input node, size:" << tensor_size
                 << " node:" << item->fullname_with_scope() << " debug:" << item->DebugString() << " index: " << index;
    if (!graph.has_flag(kFlagEnableZeroCopyInGraph) &&
        mem_manager_->MallocMem(kStaticMem, tensor_size, *device_address, graph.graph_id()) == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << tensor_size;
    }
  }
}

void KernelRuntime::AssignStaticMemoryOutput(const session::KernelGraph &graph) {
  if (graph.need_inline()) {
    return;
  }
  MS_LOG(INFO) << "AssignStaticMemoryOutput start for graph " << graph.graph_id();
  auto nodes = common::AnfAlgo::GetAllOutput(graph.output(), {prim::kPrimTupleGetItem});
  std::vector<session::KernelWithIndex> non_communication_op;
  // Assign Communicate Op Memory firstly.
  for (const auto &node : nodes) {
    // Assign output address to nop node that the attribute of "skip_nop_op_addr" is false;
    auto is_skip = !common::AnfAlgo::IsNopNode(node) || common::AnfAlgo::IsNeedSkipNopOpAddr(node);
    auto kernel_with_index = common::AnfAlgo::VisitKernelWithReturnType(node, 0, is_skip);
    MS_EXCEPTION_IF_NULL(kernel_with_index.first);
    if (!kernel_with_index.first->isa<CNode>() || !AnfUtils::IsRealKernel(kernel_with_index.first)) {
      continue;
    }
    if (common::AnfAlgo::IsCommunicationOp(kernel_with_index.first)) {
      AssignCommunicationNodeMem(kStaticMem, kernel_with_index.first);
    } else {
      non_communication_op.emplace_back(kernel_with_index);
    }
  }

  for (const auto &item_with_index : non_communication_op) {
    MS_EXCEPTION_IF_NULL(item_with_index.first);
    MS_LOG(DEBUG) << "AssignNodeOutputMem for " << item_with_index.first->fullname_with_scope();
    AssignNodeOutputMem(kStaticMem, item_with_index.first, SizeToInt(item_with_index.second));
  }
  MS_LOG(INFO) << "AssignStaticMemoryOutput end";
}

void KernelRuntime::UpdateSingleRefNodeMem(const CNodePtr &kernel, const session::KernelGraph &graph,
                                           bool reverse) const {
  MS_EXCEPTION_IF_NULL(kernel);
  auto output_num = AnfAlgo::GetOutputTensorNum(kernel);
  if (output_num == 0) {
    MS_LOG(DEBUG) << "This kernel has no output size.";
    return;
  }
  for (size_t i = 0; i < output_num; ++i) {
    session::AnfWithOutIndex out_pair(kernel, i);
    if (graph.IsInRefOutputMap(out_pair)) {
      auto origin_pair = graph.GetRefCorrespondOutput(out_pair);
      MS_EXCEPTION_IF_NULL(origin_pair.first);
      auto origin_node_output_addr = AnfAlgo::GetMutableOutputAddr(origin_pair.first, origin_pair.second);
      MS_EXCEPTION_IF_NULL(origin_node_output_addr);
      auto cur_node_output_addr = AnfAlgo::GetMutableOutputAddr(kernel, i);
      if (!reverse && origin_node_output_addr->GetPtr() == nullptr) {
        continue;
      }
      if (origin_node_output_addr.get() != cur_node_output_addr.get()) {
        MS_LOG(DEBUG) << "REF address is not same, ref node output need address update";
        MS_LOG(DEBUG) << "REF origin op is " << origin_pair.first->DebugString() << ", output index is "
                      << origin_pair.second << ", cur op is " << kernel->DebugString() << ", out index is " << i;
        if (reverse) {
          AnfAlgo::SetOutputAddr(cur_node_output_addr, origin_pair.second, origin_pair.first.get());
        } else {
          if (!cur_node_output_addr->host_shape().empty()) {
            origin_node_output_addr->set_host_shape(cur_node_output_addr->host_shape());
          }
          AnfAlgo::SetOutputAddr(origin_node_output_addr, i, kernel.get());
        }
      }
    }
  }
}

void KernelRuntime::UpdateRefNodeOutputMem(const session::KernelGraph &graph) const {
  auto &kernels = graph.execution_order();
  for (auto &kernel : kernels) {
    UpdateSingleRefNodeMem(kernel, graph, false);
  }
  for (auto it = kernels.rbegin(); it != kernels.rend(); ++it) {
    auto &kernel = *it;
    UpdateSingleRefNodeMem(kernel, graph, true);
  }
}

void KernelRuntime::AssignCommunicationNodeMem(MemType type, const AnfNodePtr &node) {
  if (!reuse_communication_address_.empty()) {
    type = kDynamicMem;
  }
  AssignCommunicationNodeInputMem(type, node);
  AssignCommunicationNodeOutputMem(type, node);
  AssignWorkSpaceMem(type, node);
}

void KernelRuntime::AssignCommunicationNodeOutputMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    MS_LOG(INFO) << "This kernel[" << node->DebugString() << "] has no output size.";
    return;
  }
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  size_t total_size = 0;
  size_t output_index = 0;
  std::vector<size_t> align_size_list;
  for (uint64_t mem_size : output_sizes) {
    if (AnfAlgo::OutputAddrExist(node, output_index++)) {
      MS_LOG(INFO) << "Communication op " << node->fullname_with_scope() << " has output device address";
      return;
    }
    if (context_ptr->get_param<bool>(MS_CTX_ENABLE_HCCL)) {
      mem_size = MemoryManager::GetCommonAlignSize(mem_size);
    }
    total_size += mem_size;
    align_size_list.emplace_back(mem_size);
  }

  if (align_size_list.empty()) {
    return;
  }

  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  uint8_t *output_ptr = nullptr;
  int64_t valid_reuse_index = -1;
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, cnode)) {
    auto reuse_index = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrReuseCommunication);
    auto it = reuse_communication_address_.find(reuse_index);
    if (it != reuse_communication_address_.end()) {
      valid_reuse_index = reuse_index;
      output_ptr = it->second.second;
    }
  }

  for (size_t j = 0; j < align_size_list.size(); ++j) {
    std::string output_format = AnfAlgo::GetOutputFormat(node, j);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, j);
    auto address = CreateDeviceAddress(nullptr, output_sizes[j], output_format, output_type, {node, j});
    MS_EXCEPTION_IF_NULL(address);
    if (output_ptr == nullptr) {
      output_ptr = mem_manager_->MallocOutputMem(node, 0, type, total_size, address, true);
      MS_EXCEPTION_IF_NULL(output_ptr);
      if (valid_reuse_index != -1) {
        auto &it = reuse_communication_address_[valid_reuse_index];
        it.second = output_ptr;
      }
    } else {
      address->set_ptr(output_ptr);
    }
    address->set_host_shape(trans::GetRuntimePaddingShape(node, j));
    AnfAlgo::SetOutputAddr(address, j, node.get());
    output_ptr += align_size_list[j];
  }
}
bool KernelRuntime::KernelMemNotReuse(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  return false;
}

DeviceAddressPtr KernelRuntime::PreAssignCNodeMemory(const AnfNodePtr &anf_node, size_t index) const {
  MS_EXCEPTION_IF_NULL(anf_node);
  if (common::AnfAlgo::IsNopNode(anf_node)) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(anf_node, index);
    return PreAssignCNodeMemory(input_node_with_index.first, input_node_with_index.second);
  }

  auto output_size = AnfAlgo::GetOutputTensorMemSize(anf_node, index);
  std::string output_format = AnfAlgo::GetOutputFormat(anf_node, index);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(anf_node, index);
  auto address = CreateDeviceAddress(nullptr, output_size, output_format, output_type, {anf_node, index});
  AnfAlgo::SetOutputAddr(address, index, anf_node.get());
  return address;
}

void KernelRuntime::AssignCommunicationNodeInputMem(MemType type, const AnfNodePtr &node) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  size_t total_size = 0;
  std::vector<std::pair<DeviceAddressPtr, size_t>> addr_size;
  size_t input_num = common::AnfAlgo::GetInputTensorNum(node);
  for (size_t i = 0; i < input_num; ++i) {
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i, true);
    auto input_node = input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    if (AnfAlgo::OutputAddrExist(input_node, input_node_with_index.second)) {
      MS_LOG(INFO) << "Communication op " << input_node->fullname_with_scope() << " has input device address";
      return;
    }
    DeviceAddressPtr address = nullptr;

    address = PreAssignCNodeMemory(input_node, input_node_with_index.second);

    MS_EXCEPTION_IF_NULL(address);
    auto mem_size = MemoryManager::GetCommonAlignSize(address->size());
    total_size += mem_size;
    addr_size.emplace_back(address, mem_size);
  }
  if (addr_size.empty()) {
    return;
  }
  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s input.";
    }
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (cnode->inputs().size() < kMinInputSize) {
    // communication node's input should contain itself and at least on input
    MS_LOG(ERROR) << "No inputs for " << cnode->fullname_with_scope();
    return;
  }

  int64_t valid_reuse_index = -1;
  uint8_t *input_ptr = nullptr;
  if (common::AnfAlgo::HasNodeAttr(kAttrReuseCommunication, cnode)) {
    auto reuse_index = common::AnfAlgo::GetNodeAttr<int64_t>(cnode, kAttrReuseCommunication);
    auto it = reuse_communication_address_.find(reuse_index);
    if (it != reuse_communication_address_.end()) {
      valid_reuse_index = reuse_index;
      input_ptr = it->second.first;
    }
  }

  if (input_ptr == nullptr) {
    auto first_input_node = cnode->input(1);
    auto prenode_index = common::AnfAlgo::VisitKernelWithReturnType(first_input_node, 0, true);
    input_ptr = mem_manager_->MallocOutputMem(prenode_index.first, prenode_index.second, type, total_size,
                                              addr_size[0].first, true);
    if (valid_reuse_index != -1) {
      auto &it = reuse_communication_address_[valid_reuse_index];
      it.first = input_ptr;
    }
  }

  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(input_ptr);
    input_ptr += iter.second;
  }
}

void KernelRuntime::AssignNodeOutputMem(MemType type, const AnfNodePtr &node, int index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);

  if (type == kSomasReuseDynamicMem) {
    bool not_reuse = KernelMemNotReuse(node);
    if (not_reuse) {
      type = kDynamicMem;
      MS_LOG(INFO) << "Disable Memory Reuse for " << node->fullname_with_scope() << "'s output.";
    }
  }

  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  if (output_sizes.empty()) {
    return;
  }
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    if ((kGetAllOuts != index) && (SizeToInt(i) != index)) {
      continue;
    }
    if (NodeOutputDeviceAddressExist(node, i)) {
      MS_LOG(DEBUG) << "Already malloc index:" << i;
      continue;
    }
    MS_LOG(DEBUG) << "Assign Node:" << node->fullname_with_scope() << " output memory size:" << output_sizes[i];
    if (type == kStaticMem) {
      MS_LOG(INFO) << "Assign Static Memory for Output node, size:" << output_sizes[i]
                   << " node:" << node->fullname_with_scope();
    }
    std::string output_format = AnfAlgo::GetOutputFormat(node, i);
    auto output_type = AnfAlgo::GetOutputDeviceDataType(node, i);
    auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type, {node, i});
    MS_EXCEPTION_IF_NULL(device_address);

    // In subgraph sink mode, graph output should not allocate memory.
    if (IsNeedAllocMem(node, i)) {
      uint8_t *ptr = mem_manager_->MallocOutputMem(node, i, type, output_sizes[i], device_address, false);
      if (ptr == nullptr && type == kSomasReuseDynamicMem) {
        MS_LOG(INFO) << "node: " << node->fullname_with_scope() << " could be a RefNode, please check it"
                     << " output index: " << i << " memory type: " << type;
      } else {
        MS_EXCEPTION_IF_NULL(ptr);
      }
    } else {
      MS_LOG(DEBUG) << "Skip mem alloc for device address:" << device_address << " node:" << node->DebugString();
    }
    device_address->set_host_shape(trans::GetRuntimePaddingShape(node, i));
    AnfAlgo::SetOutputAddr(device_address, i, node.get());
  }
}

DeviceAddressPtr KernelRuntime::AssignExtraStaticMem(const TensorPtr &tensor, const AnfNodePtr &node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
  MS_LOG(DEBUG) << "Assign Node:" << node->fullname_with_scope()
                << "Assign Static Memory for Output node, size:" << tensor_address->size();
  auto device_address = CreateDeviceAddress(nullptr, tensor_address->size(), tensor_address->format(),
                                            tensor_address->type_id(), {node, index});
  MS_EXCEPTION_IF_NULL(device_address);
  uint8_t *ptr = mem_manager_->MallocOutputMem(node, index, kStaticMem, tensor_address->size(), device_address, false);
  MS_EXCEPTION_IF_NULL(ptr);
  return device_address;
}

void KernelRuntime::AssignValueNodeTensor(const ValueNodePtr &value_node, const ValuePtr &node_value,
                                          size_t output_idx) {
  MS_EXCEPTION_IF_NULL(value_node);
  MS_EXCEPTION_IF_NULL(node_value);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::vector<tensor::TensorPtr> tensors;
  TensorValueToTensor(node_value, &tensors);
  // Graph id should be passed to record static memory if profiling is enabled.
  auto kernel_info = dynamic_cast<device::KernelInfo *>(value_node->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  uint32_t graph_id = kernel_info->graph_id();
  for (const auto &tensor : tensors) {
    if (tensor == nullptr) {
      MS_LOG(WARNING) << "Tensor is null";
      return;
    }
    auto output_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    if (output_address != nullptr && output_address->GetDeviceType() == GetTargetDeviceType()) {
      AnfAlgo::SetOutputAddr(std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address()), output_idx++,
                             value_node.get());
      continue;
    }
    size_t tensor_size = LongToSize(tensor->data().nbytes());
    auto node_size = AnfAlgo::GetOutputTensorMemSize(value_node, output_idx);
    TypeId output_type_id = AnfAlgo::GetOutputDeviceDataType(value_node, output_idx);
    if (output_type_id == kTypeUnknown) {
      output_type_id = common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
    auto output_format = AnfAlgo::GetOutputFormat(value_node, output_idx);
    DeviceAddressPtr address =
      CreateDeviceAddress(nullptr, node_size, output_format, output_type_id, {value_node, output_idx});
    address->set_host_shape(trans::GetRuntimePaddingShape(value_node, output_idx));
    address->set_from_persistent_mem(true);
    MS_EXCEPTION_IF_NULL(address);
    if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
        !mem_manager_->MallocMemFromMemPool(address, node_size)) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << node_size;
    } else {
      MS_LOG(INFO) << "Assign Static Memory for Value node, size:" << node_size
                   << " node:" << value_node->fullname_with_scope();
      if (mem_manager_->MallocMem(kStaticMem, node_size, address, graph_id) == nullptr) {
        MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << node_size;
      }
    }
    AnfAlgo::SetOutputAddr(address, output_idx, value_node.get());
    if (!address->SyncHostToDevice(trans::GetRuntimePaddingShape(value_node, 0), tensor_size, tensor->data_type(),
                                   tensor->data_c(), tensor->device_info().host_format_)) {
      MS_EXCEPTION(NotExistsError) << "ValueNode SyncHostToDevice fail!" << value_node->DebugString()
                                   << "node format is" << AnfAlgo::GetOutputFormat(value_node, output_idx)
                                   << "node dtype is "
                                   << common::AnfAlgo::GetOutputInferDataType(value_node, output_idx);
    }
  }
}

void KernelRuntime::AssignStaticMemoryValueNode(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_LOG(DEBUG) << "AssignStaticMemoryValueNode start for graph " << graph.graph_id();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  // order the value nodes
  std::map<std::string, ValueNodePtr> value_nodes_map;
  for (auto &node : graph.graph_value_nodes()) {
    MS_EXCEPTION_IF_NULL(node);
    value_nodes_map[node->fullname_with_scope()] = node;
  }

  for (auto &item : value_nodes_map) {
    auto value_node = item.second;
    MS_EXCEPTION_IF_NULL(value_node);
    if (NodeOutputDeviceAddressExist(value_node, 0)) {
      MS_LOG(DEBUG) << "value_node[" << value_node->DebugString() << "] address already exist";
      auto device_address = AnfAlgo::GetMutableOutputAddr(value_node, 0);
      if (device_address->ptr_ == nullptr) {
        if (ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER)) {
          if (!mem_manager_->MallocMemFromMemPool(device_address, device_address->size_)) {
            MS_LOG(EXCEPTION) << "MallocMemFromMemPool failed";
          }
        } else {
          if (mem_manager_->MallocMem(kStaticMem, device_address->size_, device_address, graph.graph_id())) {
            MS_LOG(EXCEPTION) << "MallocStaticMem failed";
          }
        }
      }
      continue;
    }
    auto &node_value = value_node->value();
    MS_EXCEPTION_IF_NULL(node_value);
    MS_LOG(DEBUG) << "Malloc memory for " << value_node->fullname_with_scope();
    if (node_value->isa<Tensor>() || node_value->isa<ValueTuple>()) {
      AssignValueNodeTensor(value_node, node_value, 0);
    } else if (node_value->isa<StringImm>()) {
      const bool use_mem_from_memory_pool = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) ||
                                            ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
      auto address = CreateDeviceAddressForStringValue(node_value, use_mem_from_memory_pool, graph.graph_id());
      MS_EXCEPTION_IF_NULL(address);
      address->set_from_persistent_mem(true);
      AnfAlgo::SetOutputAddr(address, 0, value_node.get());
    }
  }
  MS_LOG(DEBUG) << "AssignStaticMemoryValueNode end";
}

DeviceAddressPtr KernelRuntime::CreateDeviceAddressForStringValue(const ValuePtr &value, bool use_mem_pool,
                                                                  uint32_t graph_id) {
  auto value_string = GetValue<std::string>(value);
  size_t tensor_size = value_string.size();
  DeviceAddressPtr address = CreateDeviceAddress(nullptr, tensor_size, kOpFormat_DEFAULT, kNumberTypeUInt8);
  MS_EXCEPTION_IF_NULL(address);
  address->set_from_persistent_mem(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (use_mem_pool && !mem_manager_->MallocMemFromMemPool(address, tensor_size)) {
    MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << tensor_size;
  } else {
    MS_LOG(INFO) << "Assign Static Memory for string Value node, size:" << tensor_size;
    if (mem_manager_->MallocMem(kStaticMem, tensor_size, address, graph_id) == nullptr) {
      MS_LOG(EXCEPTION) << "Cannot alloc address when flag is: " << kStaticMem << ", tensor size is: " << tensor_size;
    }
  }
  ShapeVector shape = {1, SizeToLong(tensor_size)};
  if (!address->SyncHostToDevice(shape, tensor_size, kNumberTypeUInt8, value_string.data(), "DefaultFormat")) {
    MS_LOG(EXCEPTION) << "kValueNode SyncHostToDevice fail!";
  }
  return address;
}

bool KernelRuntime::MemSchedulerPreCompute(const AnfNodePtr &kernel, const std::shared_ptr<MemScheduler> &mem_scheduler,
                                           void *stream, bool mock, KernelLaunchInfo *kernel_launch_info) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  MS_EXCEPTION_IF_NULL(stream);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  if (!mock && common::AnfAlgo::IsCommunicationOp(kernel) && !SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }
  bool ret = mem_scheduler->PreCompute(stream);
  if (!ret) {
    return ret;
  }
  AssignKernelAddress(mem_scheduler, kernel, kernel_launch_info);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (mock && common::AnfAlgo::HasNodeAttr(kAttrOffload, cnode) &&
      common::AnfAlgo::GetNodeAttr<bool>(cnode, kAttrOffload)) {
    for (size_t i = 0; i < kernel_mod->GetOutputSizeList().size(); ++i) {
      auto device_address = AnfAlgo::GetOutputAddr(kernel, i, true);
      mem_scheduler->SetOffload(device_address);
    }
  }
  return true;
}

bool KernelRuntime::MemSchedulerPostCompute(const session::KernelGraph &graph, const AnfNodePtr &kernel,
                                            const std::shared_ptr<MemScheduler> &mem_scheduler, void *stream,
                                            bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  MS_EXCEPTION_IF_NULL(stream);
  if (!mock) {
    SyncNodeOutputTensors(mem_scheduler, graph, kernel);
  }
  bool ret = mem_scheduler->PostCompute(stream);
  if (!ret) {
    return ret;
  }
  if (!mock && common::AnfAlgo::IsCommunicationOp(kernel) && !SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed";
    return false;
  }
  return true;
}

void KernelRuntime::AssignDynamicMemory(const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_mem_reuse = EnvConfigParser::GetInstance().GetSysMemreuse();
  auto mem_type = kDynamicMem;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  if (dump_json_parser.e2e_dump_enabled() && dump_json_parser.dump_mode() == 0) {
    mindspore::EnvConfigParser::GetInstance().SetSysMemreuse(false);
    is_enable_mem_reuse = false;
    MS_LOG(INFO) << "Disable Memory Reuse when e2e dump is enable and dump mode is set to dump all kernels";
  }

  if (is_enable_mem_reuse) {
    MS_LOG(INFO) << "Memory Reuse is enable...";
    mem_manager_->MallocSomasDynamicMem(graph);
    mem_type = kSomasReuseDynamicMem;
  } else {
    MS_LOG(INFO) << "Memory Reuse is disable...";
  }
  auto &execution_nodes = graph.execution_order();
  std::vector<CNodePtr> compute_nodes;
  // communication nodes first
  for (auto &node : execution_nodes) {
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      // skip if the memory is already allocated
      AssignCommunicationNodeMem(mem_type, node);
    } else {
      compute_nodes.emplace_back(node);
    }
  }

  // then compute nodes
  for (auto &node : compute_nodes) {
    AssignNodeOutputMem(mem_type, node, kGetAllOuts);
    AssignWorkSpaceMem(mem_type, node);
  }
}

void KernelRuntime::AssignWorkSpaceMem(MemType type, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(node);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t index = 0;
  for (auto &size : kernel_mod->GetWorkspaceSizeList()) {
    if (AnfAlgo::WorkspaceAddrExist(node, index)) {
      MS_LOG(INFO) << "Op " << node->fullname_with_scope() << " has workspace device address";
      return;
    }
    auto ptr = mem_manager_->MallocWorkSpaceMem(node, index, type, size);
    AnfAlgo::SetWorkspaceAddr(CreateDeviceAddress(ptr, size, "", kTypeUnknown), index, node.get());
    index++;
  }
}

void KernelRuntime::GenLaunchArgs(const mindspore::kernel::KernelMod &kernel_mod, const mindspore::AnfNodePtr &kernel,
                                  KernelLaunchInfo *kernel_launch_info) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_launch_info);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_name = common::AnfAlgo::GetCNodeName(cnode);
  if (cnode_name == kAtomicAddrCleanOpName || cnode_name == kDynamicAtomicAddrCleanOpName) {
    return GenAddrCleanLaunchArgs(cnode, &(kernel_launch_info->inputs_));
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto skip_nop_node = (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_num; ++i) {
    if (common::AnfAlgo::IsNoneInput(kernel, i)) {
      continue;
    }
    auto real_input = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, i);
    auto device_address = AnfAlgo::GetPrevNodeOutputAddr(kernel, real_input, skip_nop_node);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(input->addr);
    input->size = device_address->size_;
    (void)kernel_launch_info->inputs_.emplace_back(input);
  }

  for (size_t i = 0; i < kernel_mod.GetOutputSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, i, skip_nop_node);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(output->addr);
    output->size = device_address->size_;
    (void)kernel_launch_info->outputs_.emplace_back(output);
  }

  for (size_t i = 0; i < kernel_mod.GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_address->ptr_;
    MS_EXCEPTION_IF_NULL(workspace->addr);
    workspace->size = device_address->size_;
    (void)kernel_launch_info->workspaces_.emplace_back(workspace);
  }
}

bool KernelRuntime::UseMemScheduler() {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!context_ptr->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    return false;
  }
  // Not use MemScheduler when running single op
  return (!context_ptr->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER) &&
          (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode));
}

void KernelRuntime::GenKernelEvents(const session::KernelGraph &graph) {
  auto &kernels = graph.execution_order();
  if (kernels.empty() || graph_kernel_events_map_.find(graph.graph_id()) != graph_kernel_events_map_.end()) {
    return;
  }
  auto kernel_events = std::pair<std::map<AnfNodePtr, std::vector<std::function<void()>>>,
                                 std::map<AnfNodePtr, std::vector<std::function<void()>>>>();
  auto &kernel_pre_run_events = kernel_events.first;
  auto &kernel_post_run_events = kernel_events.second;
  for (size_t i = 0; i < kernels.size(); ++i) {
    auto &kernel = kernels[i];
    if (!common::AnfAlgo::IsCommunicationOp(kernel)) {
      continue;
    }
    auto pre_event = CreateDeviceEvent();
    auto post_event = CreateDeviceEvent();
    MS_EXCEPTION_IF_NULL(pre_event);
    MS_EXCEPTION_IF_NULL(post_event);
    pre_event->set_wait_stream(communication_stream_);
    pre_event->set_record_stream(stream_);
    post_event->set_wait_stream(stream_);
    post_event->set_record_stream(communication_stream_);
    (void)kernel_pre_run_events[kernel].emplace_back([pre_event]() {
      pre_event->RecordEvent();
      pre_event->WaitEvent();
    });
    (void)kernel_post_run_events[kernel].emplace_back([post_event]() { post_event->RecordEvent(); });
    bool found_nearest_child = false;
    for (size_t j = i + 1; j < kernels.size(); ++j) {
      auto &child = kernels[j];
      MS_EXCEPTION_IF_NULL(child);
      if (common::AnfAlgo::IsCommunicationOp(child)) {
        continue;
      }
      auto input_size = child->inputs().size() - 1;
      for (size_t k = 0; k < input_size; ++k) {
        auto kernel_index =
          common::AnfAlgo::VisitKernelWithReturnType(common::AnfAlgo::GetInputNode(child, k), 0, true);
        if (kernel_index.first == kernel) {
          found_nearest_child = true;
          break;
        }
      }
      if (found_nearest_child) {
        (void)kernel_pre_run_events[child].emplace_back([post_event]() { post_event->WaitEvent(); });
        break;
      }
    }
    if (!found_nearest_child) {
      (void)kernel_post_run_events[kernel].emplace_back([post_event]() { post_event->WaitEvent(); });
    }
  }
  graph_kernel_events_map_[graph.graph_id()] = std::move(kernel_events);
}

void KernelRuntime::GenAddrCleanLaunchArgs(const CNodePtr &cnode, AddressPtrList *kernel_inputs,
                                           const std::shared_ptr<MemScheduler> &mem_scheduler) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  if (cnode->inputs().size() != kAtomicCleanInputSize) {
    MS_LOG(EXCEPTION) << "Atomic Addr clean Node Input nodes not equal 2.";
  }
  MS_EXCEPTION_IF_NULL(cnode->inputs()[1]);
  auto pre_node = (cnode->inputs()[1])->cast<CNodePtr>();
  // set clean output address
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicOutputIndexs, pre_node)) {
#if defined(__APPLE__)
    auto clean_output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<int>>(pre_node, kAttrAtomicOutputIndexs);
#else
    auto clean_output_indexes = common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicOutputIndexs);
#endif
    for (auto index : clean_output_indexes) {
      auto device_address = AnfAlgo::GetOutputAddr(pre_node, index);
      kernel::AddressPtr input = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(input);
      if (mem_scheduler != nullptr) {
        GetOrMallocAddress(mem_scheduler, device_address, input);
      } else {
        input->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(input->addr);
      }
      auto real_output_size = AnfAlgo::GetOutputTensorMemSize(pre_node, index);
      input->size = device_address->size_;
      if (device_address->size_ != real_output_size) {
        MS_LOG(DEBUG) << "The node:" << pre_node->fullname_with_scope() << " real output size is " << real_output_size;
        input->size = real_output_size;
      }
      kernel_inputs->emplace_back(input);
    }
    MS_LOG(DEBUG) << "AtomicAddClean clean output size:" << clean_output_indexes.size();
  }
  // set clean workspace address
  if (common::AnfAlgo::HasNodeAttr(kAttrAtomicWorkspaceIndexs, pre_node)) {
#if defined(__APPLE__)
    auto clean_workspaces_indexes =
      common::AnfAlgo::GetNodeAttr<std::vector<int>>(pre_node, kAttrAtomicWorkspaceIndexs);
#else
    auto clean_workspaces_indexes =
      common::AnfAlgo::GetNodeAttr<std::vector<size_t>>(pre_node, kAttrAtomicWorkspaceIndexs);
#endif
    for (const auto &index : clean_workspaces_indexes) {
      auto device_address = AnfAlgo::GetWorkspaceAddr(pre_node, index);
      kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
      MS_EXCEPTION_IF_NULL(workspace);
      if (mem_scheduler != nullptr) {
        GetOrMallocAddress(mem_scheduler, device_address, workspace);
      } else {
        workspace->addr = device_address->ptr_;
        MS_EXCEPTION_IF_NULL(workspace->addr);
      }
      workspace->size = device_address->size_;
      kernel_inputs->emplace_back(workspace);
    }
  }
}

void KernelRuntime::LaunchKernelEvent(const std::map<AnfNodePtr, std::vector<std::function<void()>>> &kernel_events,
                                      const AnfNodePtr &node) const {
  if (kernel_events.find(node) == kernel_events.end()) {
    return;
  }

  for (auto &event : kernel_events.at(node)) {
    event();
  }
}

bool KernelRuntime::LaunchKernelWithPynativeProfiling(kernel::KernelMod *kernel_mod, const std::string &op_name,
                                                      const KernelLaunchInfo &kernel_launch_info, void *stream) {
  MS_EXCEPTION_IF_NULL(kernel_mod);
  MS_EXCEPTION_IF_NULL(stream);
  float cost_time = 0;
  auto start = CreateDeviceTimeEvent();
  auto end = CreateDeviceTimeEvent();
  MS_EXCEPTION_IF_NULL(start);
  MS_EXCEPTION_IF_NULL(end);
  start->set_record_stream(stream);
  end->set_record_stream(stream);
  start->RecordEvent();
  bool ret = kernel_mod->Launch(kernel_launch_info, stream);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name is : " << op_name;
  }
  end->RecordEvent();
  start->SyncEvent();
  end->SyncEvent();
  start->ElapsedTime(&cost_time, end.get());
  MS_LOG(DEBUG) << "Launch kernel:" << op_name << " cost:" << cost_time / kBasicTimeTransferUnit;
  return ret;
}

void KernelRuntime::DebugStreamSync(const CNodePtr &kernel) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto enable_sync_run = ms_context->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_SYNCHRONIZE);
  if (enable_sync_run) {
    if (!SyncStream()) {
      MS_LOG(EXCEPTION) << "Op " << kernel->fullname_with_scope() << " run failed!";
    }
  }
}

void KernelRuntime::GetOrMallocAddress(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                       const DeviceAddress *device_address, const kernel::AddressPtr &kernel_addr) {
  if (device_address->ptr_ != nullptr) {
    kernel_addr->addr = device_address->ptr_;
  } else {
    kernel_addr->addr = mem_scheduler->GetOrMalloc(device_address, device_address->size_);
  }
}

void KernelRuntime::AssignKernelAddress(const std::shared_ptr<MemScheduler> &mem_scheduler, const AnfNodePtr &kernel,
                                        KernelLaunchInfo *kernel_launch_info) const {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_launch_info);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto cnode_name = common::AnfAlgo::GetCNodeName(cnode);
  if (cnode_name == kAtomicAddrCleanOpName || cnode_name == kDynamicAtomicAddrCleanOpName) {
    return GenAddrCleanLaunchArgs(cnode, &(kernel_launch_info->inputs_), mem_scheduler);
  }
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  const auto update_parameter = common::AnfAlgo::IsUpdateParameterKernel(cnode);
  for (size_t j = 0; j < input_num; ++j) {
    auto real_input = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, j);
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel, real_input, true);
    auto index = kernel_with_index.second;
    auto &input_node = kernel_with_index.first;
    auto device_address = AnfAlgo::GetOutputAddr(input_node, index, true);
    MS_EXCEPTION_IF_NULL(device_address);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, input);
    input->size = device_address->size_;
    (void)kernel_launch_info->inputs_.emplace_back(input);
    if (update_parameter && input_node->isa<Parameter>()) {
      auto param = input_node->cast<ParameterPtr>();
      auto abstract = param->abstract();
      MS_EXCEPTION_IF_NULL(abstract);
      if (abstract->isa<abstract::AbstractRefTensor>()) {
        mem_scheduler->UpdateHighPriorityMem(device_address);
      }
    }
  }

  for (size_t j = 0; j < kernel_mod->GetOutputSizeList().size(); ++j) {
    auto device_address = AnfAlgo::GetOutputAddr(kernel, j, true);
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, output);
    output->size = device_address->size_;
    (void)kernel_launch_info->outputs_.emplace_back(output);
  }

  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetWorkspaceAddr(kernel, i);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    GetOrMallocAddress(mem_scheduler, device_address, workspace);
    workspace->size = device_address->size_;
    (void)kernel_launch_info->workspaces_.emplace_back(workspace);
  }
}

void KernelRuntime::SyncNodeOutputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                          const session::KernelGraph &graph, const AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t input_idx = 0; input_idx < kernel_mod->GetInputSizeList().size(); ++input_idx) {
    const auto input_node_index = common::AnfAlgo::GetPrevNodeOutput(kernel, input_idx, true);
    if (input_node_index.first != nullptr && input_node_index.first->isa<Parameter>()) {
      SyncNodeOutputTensor(mem_scheduler, input_node_index, graph);
    }
  }
  for (size_t output_idx = 0; output_idx < kernel_mod->GetOutputSizeList().size(); ++output_idx) {
    SyncNodeOutputTensor(mem_scheduler, std::make_pair(kernel, output_idx), graph);
  }
}

void KernelRuntime::SyncNodeOutputTensor(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                         const KernelWithIndex &node_output_index, const session::KernelGraph &graph) {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  if (node_output_index.first == nullptr) {
    return;
  }
  auto device_address = AnfAlgo::GetMutableOutputAddr(node_output_index, true);
  auto tensor = graph.GetNodeOutputTensor(node_output_index);
  if (tensor == nullptr) {
    return;
  }
  if (device_address == nullptr) {
    tensor->data_sync(false);
    tensor->set_device_address(nullptr);
    tensor->set_sync_status(kNeedSyncHostToDevice);
    return;
  }
  if (!SyncStream()) {
    MS_LOG(EXCEPTION) << "SyncStream failed";
  }
  auto origin_ptr = device_address->ptr_;
  if (device_address->ptr_ == nullptr) {
    device_address->ptr_ = mem_scheduler->GetOrMalloc(device_address.get(), device_address->size_);
  }
  tensor->set_device_address(device_address);
  tensor->data_sync(false);
  tensor->set_device_address(nullptr);
  device_address->ptr_ = origin_ptr;
  tensor->set_sync_status(kNeedSyncHostToDevice);
}

void KernelRuntime::InitGraphInputTensors(const std::shared_ptr<MemScheduler> &mem_scheduler,
                                          const session::KernelGraph &graph) const {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  auto &input_nodes = graph.input_nodes();
  auto &input_tensors = graph.input_tensors();
  if (input_tensors.size() != input_nodes.size()) {
    MS_LOG_EXCEPTION << "Invalid input tensor size:" << input_tensors.size() << " vs node size:" << input_nodes.size();
  }
  mem_scheduler->ClearMemNeedInit();
  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto input_node = input_nodes[i];
    if (!input_node->isa<Parameter>() || !AnfAlgo::OutputAddrExist(input_node, 0)) {
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    auto tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    auto tensor_address = std::dynamic_pointer_cast<device::DeviceAddress>(tensor->device_address());
    const auto tensor_size = LongToSize(tensor->data().nbytes());
    bool need_sync = false;
    if (tensor->NeedSyncHostToDevice()) {
      need_sync = true;
    } else if (tensor_address != device_address) {
      tensor->data_sync(false);
      need_sync = true;
    }
    if (mem_scheduler->HasDeviceMem(device_address.get())) {
      device_address->set_ptr(nullptr);
    }
    if (need_sync) {
      const auto &shape = trans::GetRuntimePaddingShape(input_node, 0);
      if (device_address->GetPtr() != nullptr) {
        (void)device_address->SyncHostToDevice(shape, LongToSize(tensor->data().nbytes()), tensor->data_type(),
                                               tensor->data_c(), tensor->device_info().host_format_);
      } else {
        mem_scheduler->AddMemNeedInit(device_address.get());
      }
    }
    MemPriority priority = kMemPriorityLow;
    const auto &parameter = input_node->cast<ParameterPtr>();
    if (common::AnfAlgo::IsParameterWeight(parameter) || graph.IsUpdatedParameter(parameter)) {
      priority = kMemPriorityHigh;
    }
    mem_scheduler->Init(device_address.get(), tensor->data_c(), tensor_size, priority);
    tensor->set_sync_status(kNoNeedSync);
  }
}

void KernelRuntime::AddCommunicationMemInfo(const session::KernelGraph &graph) {
  const auto mem_scheduler = mem_scheduler_manager_.GetOrCreateMemScheduler(graph.graph_id());
  for (size_t compute_index = 0; compute_index < graph.execution_order().size(); ++compute_index) {
    const auto &kernel = graph.execution_order()[compute_index];
    MS_EXCEPTION_IF_NULL(kernel);
    if (!common::AnfAlgo::IsCommunicationOp(kernel)) {
      continue;
    }
    auto device_address_to_key = [](const DeviceAddressPtr &device_address) -> void * { return device_address.get(); };
    size_t input_total_size = 0;
    DeviceAddressPtrList input_address_list;
    std::vector<size_t> input_align_size_list;
    GetCommunicationInputInfo(kernel, &input_total_size, &input_address_list, &input_align_size_list);
    if (input_address_list.size() > 1) {
      std::vector<const void *> input_address_key_list;
      (void)std::transform(input_address_list.begin(), input_address_list.end(),
                           std::back_inserter(input_address_key_list), device_address_to_key);
      mem_scheduler->AddContinuousMemInfo(true, compute_index, input_total_size, input_align_size_list,
                                          input_address_key_list);
    }
    size_t output_total_size = 0;
    DeviceAddressPtrList output_address_list;
    std::vector<size_t> output_align_size_list;
    GetCommunicationOutputInfo(kernel, &output_total_size, &output_address_list, &output_align_size_list);
    if (output_address_list.size() > 1) {
      std::vector<const void *> output_address_key_list;
      (void)std::transform(output_address_list.begin(), output_address_list.end(),
                           std::back_inserter(output_address_key_list), device_address_to_key);
      mem_scheduler->AddContinuousMemInfo(false, compute_index, output_total_size, output_align_size_list,
                                          output_address_key_list);
    }
  }
}

bool KernelRuntime::LaunchKernel(const session::KernelGraph &graph, const AnfNodePtr &kernel,
                                 const std::shared_ptr<MemScheduler> &mem_scheduler, bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  KernelLaunchInfo kernel_launch_info;
  auto stream = GetKernelStream(kernel);
  MS_EXCEPTION_IF_NULL(stream);
  bool ret = true;
  if (mem_scheduler != nullptr) {
    ret = MemSchedulerPreCompute(kernel, mem_scheduler, stream, mock, &kernel_launch_info);
    if (!ret) {
      return ret;
    }
  } else if (!kernel_mod->GetInputsAddr().empty() || !kernel_mod->GetOutputsAddr().empty()) {
    kernel_launch_info.inputs_ = kernel_mod->GetInputsAddr();
    kernel_launch_info.outputs_ = kernel_mod->GetOutputsAddr();
    kernel_launch_info.workspaces_ = kernel_mod->GetWorkSpacesAddr();
  } else {
    GenLaunchArgs(*kernel_mod, kernel, &kernel_launch_info);
  }
  if (!mock) {
    if (pynative_mode_profiling_flag_) {
      ret = LaunchKernelWithPynativeProfiling(kernel_mod, kernel->fullname_with_scope(), kernel_launch_info, stream);
    } else {
      ret = kernel_mod->Launch(kernel_launch_info, stream);
    }
    if (!ret) {
      return ret;
    }
  }
  if (mem_scheduler != nullptr) {
    ret = MemSchedulerPostCompute(graph, kernel, mem_scheduler, stream, mock);
  }
  return ret;
}

bool KernelRuntime::LaunchKernelMod(const session::KernelGraph &graph, bool mock) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::shared_ptr<MemScheduler> mem_scheduler = nullptr;

  if (UseMemScheduler()) {
    mem_scheduler = mem_scheduler_manager_.GetOrCreateMemScheduler(graph.graph_id());
    MS_EXCEPTION_IF_NULL(mem_scheduler);
    mem_scheduler->Reset();
    mem_scheduler->Update();
    InitGraphInputTensors(mem_scheduler, graph);
  }

  const auto &kernels = graph.execution_order();
  std::map<AnfNodePtr, std::vector<std::function<void()>>> kernel_pre_run_events;
  std::map<AnfNodePtr, std::vector<std::function<void()>>> kernel_post_run_events;
  auto events_iter = graph_kernel_events_map_.find(graph.graph_id());
  if (events_iter != graph_kernel_events_map_.end()) {
    kernel_pre_run_events = events_iter->second.first;
    kernel_post_run_events = events_iter->second.second;
  }
  for (size_t i = 0; i < kernels.size(); ++i) {
    LaunchKernelEvent(kernel_pre_run_events, kernels[i]);
    auto &kernel = kernels[i];
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
      MS_EXCEPTION_IF_NULL(kernel_mod);
      opt::dynamic_shape::InferOp(kernel);
      auto args = kernel::GetArgsFromCNode(kernel);
      if (kernel_mod->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
          static_cast<int>(kernel::KRET_RESIZE_FAILED)) {
        MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize  failed.";
      }
      KernelLaunchInfo kernel_launch_info;
      device::KernelRuntime::GenLaunchArgs(*kernel_mod, kernel, &kernel_launch_info);

      // allocate workspace size
      std::vector<AddressPtr> workspace_addr;
      if (AnfAlgo::GetKernelType(kernel) == KernelType::TBE_KERNEL) {
        MS_EXCEPTION_IF_NULL(tbe_call_);
        tbe_call_(kernel, kernel_mod, &workspace_addr);
      } else {
        workspace_addr = kernel_launch_info.workspaces_;
      }

      auto ret = kernel_mod->Launch(kernel_launch_info.inputs_, workspace_addr, kernel_launch_info.outputs_, stream_);
      if (!ret) {
        MS_LOG(ERROR) << "Launch kernel failed, kernel full name: " << kernel->fullname_with_scope();
        return false;
      }

      if (!SyncStream()) {
        MS_LOG(ERROR) << "SyncStream failed";
        return false;
      }
      kernel::UpdateNodeShape(kernel);
    } else {
      // Skip transpose kernel with "nop_op" attr which is not hidden or removed in PyNative infer scenario. Transpose
      // kernel, which is not supposed to be executed, is generated in TransDataSplit to support specific Transdata.
      // And hard code here should be removed after new Transdata programme is implemented in the foreseeable future.
      if (common::AnfAlgo::HasNodeAttr(kAttrNopOp, kernel)) {
        for (size_t idx = 0; idx < AnfAlgo::GetOutputTensorNum(kernel); idx += 1) {
          auto real_input = AnfAlgo::GetInputGraphIdxByKernelIdx(kernel, idx);
          auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, real_input);
          AnfAlgo::SetOutputAddr(device_address, idx, kernel.get());
        }
        continue;
      }
      auto ret = LaunchKernel(graph, kernel, mem_scheduler, mock);
      if (!ret) {
        MS_LOG(ERROR) << "Launch kernel failed.";
        return false;
      }
      KernelLaunchProfiling(kernel->fullname_with_scope());
      DebugStreamSync(kernel);
    }
    LaunchKernelEvent(kernel_post_run_events, kernels[i]);
  }
  if (UseMemScheduler() && !mock) {
    SyncParameter(graph, mem_scheduler);
  }
  return true;
}

void KernelRuntime::SyncParameter(const session::KernelGraph &graph,
                                  const std::shared_ptr<MemScheduler> &mem_scheduler) const {
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  auto &input_nodes = graph.input_nodes();
  auto &input_tensors = graph.input_tensors();
  if (input_tensors.size() != input_nodes.size()) {
    MS_LOG_EXCEPTION << "Invalid input tensor size:" << input_tensors.size() << " vs node size:" << input_nodes.size();
  }

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    auto input_node = input_nodes[i];
    if (!input_node->isa<Parameter>() || !AnfAlgo::OutputAddrExist(input_node, 0)) {
      continue;
    }
    auto device_address = AnfAlgo::GetMutableOutputAddr(input_node, 0);
    MS_EXCEPTION_IF_NULL(device_address);
    auto parameter = input_node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    if (!common::AnfAlgo::IsParameterWeight(parameter) && !graph.IsUpdatedParameter(parameter)) {
      continue;
    }
    auto tensor = input_tensors[i];
    MS_EXCEPTION_IF_NULL(tensor);
    if (mem_scheduler->HasDeviceMem(device_address.get())) {
      auto device_ptr = mem_scheduler->GetOrMalloc(device_address.get(), device_address->size(), kMemPriorityHigh);
      device_address->set_ptr(device_ptr);
      tensor->set_device_address(device_address);
      tensor->set_sync_status(kNeedSyncDeviceToHost);
    }
    if (graph.IsUpdatedParameter(parameter)) {
      tensor->SetIsUpdateByDevice();
    }
  }
}

void KernelRuntime::UseMemSchedulerIfNeeded(const session::KernelGraph &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (!UseMemScheduler()) {
    return;
  }
  auto mem_scheduler = mem_scheduler_manager_.GetOrCreateMemScheduler(graph.graph_id());
  MS_EXCEPTION_IF_NULL(mem_scheduler);
  if (mem_scheduler->optimized()) {
    return;
  }
  mem_scheduler->SetMemHandler(std::make_shared<MemHandler>(mem_manager_));
  mem_scheduler->SetTotalStep(graph.execution_order().size());

  if (mem_scheduler->need_record_event()) {
    (void)LaunchKernelMod(graph, true);
    mem_scheduler->set_need_record_event(false);
  }
  auto ret = mem_scheduler->Optimize();
  if (!ret) {
    MS_LOG_EXCEPTION << "Can't run graph " << graph.graph_id() << " for memory limit.";
  }
}

bool KernelRuntime::LaunchKernels(const session::KernelGraph &graph) {
  UseMemSchedulerIfNeeded(graph);
  if (!LaunchKernelMod(graph)) {
    MS_LOG(ERROR) << "LaunchKernelMod failed!";
    return false;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->get_param<int>(MS_CTX_EXECUTION_MODE) == kGraphMode) {
    if (!SyncStream()) {
      MS_LOG(ERROR) << "SyncStream failed";
      return false;
    }
  }
  return true;
}

void KernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  MS_LOG(INFO) << "Clear graph:" << graph_id << " runtime resource";
}
}  // namespace device
}  // namespace mindspore
