/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "backend/ge_backend/executor/ge_memory_allocator.h"

#include <tuple>
#include <functional>
#include <algorithm>
#include <utility>
#include <map>
#include <set>
#include <sstream>
#include <vector>
#include <memory>
#include <string>

#include "backend/ge_backend/graph_ir/utils.h"
#include "include/utils/utils.h"
#include "mindspore/ccsrc/utils/ir_dump/draw.h"
#include "mindspore/ccsrc/utils/ir_dump/anf_ir_dump.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_info.h"
#include "include/backend/common/kernel_graph/kernel_graph.h"
#include "backend/ge_backend/utils/device_address_utils.h"
#include "backend/ge_backend/executor/ge_memory_manager.h"
#include "tools/profiler/profiling.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_build_info.h"
#include "include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
namespace {
constexpr size_t kNeedRecycleOutput = 5;
using mindspore::session::KernelWithIndex;

void UpdateTracker(const std::string &task_name, const std::string &node_name, const std::string &graph_str,
                   size_t size, void *device_ptr, memory::mem_pool::MemType mem_type) {
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, task_name, node_name, graph_str, false);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, task_name, size, device_ptr, mem_type);
}

std::multimap<std::string, ParameterPtr> FilterAllParameters(const KernelGraphPtr &kernel_graph) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::multimap<std::string, ParameterPtr> ret;
  std::vector<AnfNodePtr> todo = kernel_graph->input_nodes();
  (void)todo.insert(todo.end(), kernel_graph->child_graph_result().begin(), kernel_graph->child_graph_result().end());
  for (const auto &node : todo) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<Parameter>()) {
      continue;
    }
    auto parameter = node->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(parameter);
    std::string name = parameter->name();
    (void)ret.emplace(name, parameter);
  }
  return ret;
}

void SetParameterKernelInfo(const AnfNodePtr &node, const std::shared_ptr<device::KernelInfo> &kernel_info) {
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    MS_LOG(ERROR) << "Parameter doesn't have build info: " << node->DebugString()
                  << ", full name: " << node->fullname_with_scope();
    return;
  }
  std::vector<TypeId> refresh_output_types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  build_info->SetOutputsDeviceType(refresh_output_types);
}

void SetKernelInfo(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  // If kernel build info has been set up. skip
  std::shared_ptr<device::KernelInfo> kernel_info =
    std::dynamic_pointer_cast<device::KernelInfo>(node->kernel_info_ptr());
  if (utils::isa<ParameterPtr>(node)) {
    SetParameterKernelInfo(node, kernel_info);
    return;
  }

  if (!kernel_info) {
    kernel_info = std::make_shared<device::KernelInfo>();
    MS_EXCEPTION_IF_NULL(kernel_info);
    node->set_kernel_info(kernel_info);
  }

  auto build_info = kernel_info->GetMutableSelectKernelBuildInfo();
  if (!build_info) {
    auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
    MS_EXCEPTION_IF_NULL(builder);
    build_info = builder->Build();
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(node);
  std::vector<TypeId> output_infer_types;
  std::vector<std::string> output_formats;
  for (const auto &output_with_index : output_with_indexs) {
    (void)output_infer_types.emplace_back(
      common::AnfAlgo::GetOutputInferDataType(output_with_index.first, output_with_index.second));
    (void)output_formats.emplace_back(kOpFormat_DEFAULT);
  }
  build_info->SetOutputsDeviceType(output_infer_types);
  build_info->SetOutputsFormat(output_formats);
  kernel_info->set_select_kernel_build_info(build_info);
}

KernelTensorPtr CreateOutputKernelTensor(const KernelGraphPtr &kernel_graph, const KernelWithIndex &output_with_index,
                                         size_t need_alloc_output_cnt, GeDeviceResManagerPtr res_manager) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto output_node = output_with_index.first;
  MS_EXCEPTION_IF_NULL(output_node);
  auto ref_map = kernel_graph->GetRefMap();

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto real_index = output_node->isa<ValueNode>() ? 0 : output_with_index.second;
  TypeId output_type_id = common::AnfAlgo::GetOutputInferDataType(output_node, real_index);
  size_t type_size = GetTypeByte(TypeIdToType(output_type_id));
  auto shapes = AnfAlgo::GetRuntimePaddingShape(output_node, real_index);
  auto tensor_size =
    shapes.empty() ? type_size : std::accumulate(shapes.begin(), shapes.end(), type_size, std::multiplies<size_t>());
  // When ValueNode is a graph output, runtime does not manage this memory
  // output in ref_map, mem same is input
  // ge kernel no need alloc memory
  bool need_not_alloc = (kernel_graph->has_flag(kFlagEnableZeroCopyInGraph) && !output_node->isa<ValueNode>()) ||
                        (ref_map.find(output_with_index) != ref_map.end());
  MS_EXCEPTION_IF_NULL(res_manager);
  void *mem = need_not_alloc ? nullptr : res_manager->AllocateMemory(tensor_size);

  if (memory::mem_pool::IsNeedProfilieMemoryLog() && !need_not_alloc) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ValueNodeOutput, size:" << tensor_size
                    << ", graph: " << kernel_graph->ToString() << ", node: " << output_node->fullname_with_scope()
                    << ", device address addr: " << mem;
  }
  if (!need_not_alloc) {
    UpdateTracker("ValueNodeOutput", output_node->fullname_with_scope(), kernel_graph->ToString(), tensor_size, mem,
                  memory::mem_pool::MemType::kConstantValue);
  }

  const auto &kernel_tensor = AnfAlgo::CreateOutputKernelTensorWithDeviceInfo(
    {output_node, real_index}, mem, tensor_size, kOpFormat_DEFAULT, output_type_id, {}, kAscendDevice, device_id);
  // Set kernel tensor with output_with_index.second.
  AnfAlgo::SetOutputKernelTensor(kernel_tensor, output_with_index.second, output_node.get());
  auto output_device_addr = kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(output_device_addr);
  if (ref_map.find(output_with_index) != ref_map.end()) {
    auto input_with_index = ref_map[output_with_index];
    auto input_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_with_index.first, input_with_index.second, false);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    auto input_device_address = input_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(input_device_address);
    MS_LOG(INFO) << "The output node " << output_node->fullname_with_scope()
                 << " is in ref_map, set the same device_address ptr as the corresponding input, input node: "
                 << input_with_index.first->fullname_with_scope();
    // Update the reference count of device address.
    kernel_tensor->set_pointer_ref_count(input_kernel_tensor.get());
    kernel_tensor->IncreaseOriginalRefCount();
    kernel_tensor->ResetRefCount();
  }
  if (memory::mem_pool::IsMemoryPoolRecycle() && need_alloc_output_cnt <= kNeedRecycleOutput) {
    MS_LOG(INFO) << "Set Memory Pool Recycle, graph: " << kernel_graph->ToString()
                 << ", node: " << output_node->fullname_with_scope();
    output_device_addr->set_from_persistent_mem(true);
    output_device_addr->set_need_recycle(true);
  }
  return kernel_tensor;
}

void AllocParameterMemory(const KernelGraphPtr &kernel_graph, std::set<KernelGraphPtr> *memo) {
  // Set Device Type to be same as Host Type, AssignStaticMemoryInput will ignore parameters without DeviceType
  MS_EXCEPTION_IF_NULL(kernel_graph);
  if (memo == nullptr) {
    MS_LOG(INFO) << "Start AllocParameterMemory, kernel graph: " << kernel_graph->ToString();
    std::set<KernelGraphPtr> memo_set;
    AllocParameterMemory(kernel_graph, &memo_set);
    MS_LOG(INFO) << "AllocParameterMemory finish.";
    return;
  } else if (memo->find(kernel_graph) != memo->end()) {
    return;
  }
  (void)memo->insert(kernel_graph);
  auto parameters = FilterAllParameters(kernel_graph);
  for (const auto &iter : parameters) {
    auto parameter = utils::cast<ParameterPtr>(iter.second);
    if (parameter == nullptr) {
      continue;
    }
    SetKernelInfo(parameter);
  }
  DeviceAddressUtils::CreateParameterDeviceAddress(kernel_graph);
}

void AllocOutputMemory(const KernelGraphPtr &kernel_graph, GeDeviceResManagerPtr res_manager) {
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_LOG(INFO) << "Start AllocOutputMemory, kernel graph: " << kernel_graph->ToString();

  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(kernel_graph->output());
  auto ref_map = kernel_graph->GetRefMap();
  size_t need_alloc_output_cnt = 0;
  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    SetKernelInfo(output_node);
    if (output_node->isa<Parameter>() || output_node->isa<ValueNode>()) {
      continue;
    }
    if (ref_map.find(output_with_index) != ref_map.end()) {
      continue;
    }
    need_alloc_output_cnt++;
  }

  for (const auto &output : outputs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);

    // Parameter's memory is allocated earlier, and there is no need to reallocate memory if Parameter is output.
    if (AnfAlgo::OutputAddrExist(output_node, output_with_index.second, false) || output_node->isa<Parameter>()) {
      MS_LOG(INFO) << "The device_address of output node " << output_node->fullname_with_scope()
                   << " is already exist, skip.";
      continue;
    }

    if (HasAbstractMonad(output_node)) {
      continue;
    }

    auto output_kernel_tensor =
      CreateOutputKernelTensor(kernel_graph, output_with_index, need_alloc_output_cnt, res_manager);
    auto output_device_addr = output_kernel_tensor->device_address();
    AnfAlgo::SetOutputAddr(output_device_addr, output_kernel_tensor->tensor_storage_info(), output_with_index.second,
                           output_node);
    MS_LOG(INFO) << "Output node info: (name " << output_node->fullname_with_scope() << ", "
                 << output_node->DebugString() << " ), output size: " << output_device_addr->GetSize()
                 << ", device_address: " << output_device_addr;
    // When both the input and output of NopNode are used as outputs, different memory needs to be allocated for them.
  }
  MS_LOG(INFO) << "AllocOutputMemory finish.";
}

void EnableGraphInputZeroCopy(const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // Zero copy is only enabled for PyNative and Subgraph sink.
  if ((!graph->has_flag(kFlagPyNativeRunInGraph) && !graph->has_flag(kFlagEnableZeroCopyInGraph)) ||
      !graph->is_graph_run_mode()) {
    return;
  }
  const auto &input_nodes = graph->input_nodes();
  for (const auto &input : input_nodes) {
    MS_EXCEPTION_IF_NULL(input);
    if (AnfAlgo::OutputAddrExist(input, 0)) {
      auto input_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input, 0, false);
      MS_EXCEPTION_IF_NULL(input_kernel_tensor);
      input_kernel_tensor->set_is_ptr_persisted(false);
      input_kernel_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
      MS_LOG(INFO) << "Enable zero copy for input " << input->DebugString();
    }
  }
}

void EnableGraphOutputZeroCopy(const KernelGraphPtr &graph) {
  MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start";
  MS_EXCEPTION_IF_NULL(graph);
  if ((!graph->has_flag(kFlagEnableZeroCopyInGraph)) || !graph->is_graph_run_mode()) {
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy start return";
    return;
  }
  // Zero copy is only enabled for subgraph sink.
  auto outputs = common::AnfAlgo::GetAllOutputWithIndex(graph->output());
  for (const auto &output : outputs) {
    const auto &node_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(output);
    const auto &node = node_with_index.first;
    const auto &index = node_with_index.second;
    MS_EXCEPTION_IF_NULL(node);
    MS_LOG(DEBUG) << "EnableGraphOutputZeroCopy check node:" << node->DebugString();
    if (node->isa<CNode>() && AnfAlgo::OutputAddrExist(node, index)) {
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(node, index, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      kernel_tensor->set_is_ptr_persisted(false);
      MS_LOG(DEBUG) << "Disable ptr persisted in output node:" << node->DebugString() << " index:" << index
                    << " kernel tensor:" << kernel_tensor->ToString() << " for graph:" << graph->ToString();
    }
  }
}

void AllocConstMemory(const backend::ge_backend::RunOptions &options, const KernelGraphPtr &graph, size_t memory_size,
                      GeDeviceResManagerPtr res_manager) {
  if (memory_size == 0) {
    return;
  }
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_LOG(INFO) << "Start AllocConstMemory, memory_size: " << memory_size;
  auto memory = res_manager->AllocateMemory(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "Allocate memory failed, memory size:" << memory_size << ", graph: " << graph->ToString();
  }
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: ConstMemory, size: " << memory_size
                    << ", graph: " << graph->ToString() << ", device address addr: " << memory;
  }
  UpdateTracker("AllocConstMemory", "ConstMemory", graph->ToString(), memory_size, memory,
                memory::mem_pool::MemType::kGeConst);
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->SetConstMemory(options, memory, memory_size);
  if (ret != backend::ge_backend::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "SetConstMemory for graph " << options.name << " failed.";
  }
  MS_LOG(INFO) << "End AllocConstMemory";
}

void AllocFeatureMemory(const backend::ge_backend::RunOptions &options, size_t memory_size,
                        GeDeviceResManagerPtr res_manager) {
  if (memory_size == 0) {
    return;
  }
  MS_LOG(INFO) << "Start AllocFeatureMemory, memory_size: " << memory_size;
  MS_EXCEPTION_IF_NULL(res_manager);
  auto memory = res_manager->AllocateWorkSpaceMemory(memory_size);
  if (memory == nullptr) {
    MS_LOG(EXCEPTION) << "AllocFeatureMemory error, memory not enough, memory size: " << memory_size;
  }
  auto graph_runner = backend::ge_backend::GetGraphRunner();
  MS_EXCEPTION_IF_NULL(graph_runner);
  auto ret = graph_runner->UpdateFeatureMemory(options, memory, memory_size);
  if (ret != backend::ge_backend::Status::SUCCESS) {
    MS_LOG(EXCEPTION) << "UpdateFeatureMemory for graph " << options.name << " failed.";
  }

  MS_LOG(INFO) << "End AllocFeatureMemory";
}
}  // namespace

void GEMemoryAllocator::AllocUnuseInput(const KernelGraphPtr &kernel_graph, const AnfNodePtr &input_node,
                                        device::DeviceAddress *output_addr, GeDeviceResManagerPtr res_manager) {
  std::vector<size_t> shape = Convert2SizeT(common::AnfAlgo::GetOutputInferShape(input_node, 0));
  size_t type_size = GetTypeByte(TypeIdToType(common::AnfAlgo::GetOutputInferDataType(input_node, 0)));
  size_t memory_size = std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>{});
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_EXCEPTION_IF_NULL(output_addr);
  auto memory = res_manager->AllocateMemory(memory_size);
  output_addr->set_ptr(memory);
  output_addr->SetSize(memory_size);
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: UnusedInput, size:" << memory_size
                    << ", graph: " << kernel_graph->ToString() << ", node: " << input_node->fullname_with_scope()
                    << ", device address addr: " << memory;
  }
  UpdateTracker("UnusedInput", input_node->fullname_with_scope(), kernel_graph->ToString(), memory_size, memory,
                memory::mem_pool::MemType::kOther);
}

void GEMemoryAllocator::AllocUnuseInput(const KernelGraphPtr &kernel_graph, kernel::KernelTensor *tensor,
                                        GeDeviceResManagerPtr res_manager) {
  MS_EXCEPTION_IF_NULL(res_manager);
  MS_EXCEPTION_IF_NULL(tensor);
  auto memory = res_manager->AllocateMemory(tensor->size());
  tensor->set_device_ptr(memory);
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, alloc type: UnusedInput, size:" << tensor->size()
                    << ", graph: " << kernel_graph->ToString() << ", device address addr: " << memory;
  }
  UpdateTracker("UnusedInput", kernel_graph->ToString(), kernel_graph->ToString(), tensor->size(), memory,
                memory::mem_pool::MemType::kOther);
}

void GEMemoryAllocator::ProcessGraphDeviceAddress(const KernelGraphPtr &kernel_graph,
                                                  GeDeviceResManagerPtr res_manager) {
  AllocParameterMemory(kernel_graph, nullptr);
  AllocOutputMemory(kernel_graph, res_manager);
  EnableGraphInputZeroCopy(kernel_graph);
  EnableGraphOutputZeroCopy(kernel_graph);
}

void GEMemoryAllocator::AllocGraphMemory(const backend::ge_backend::RunOptions &options, const KernelGraphPtr &graph,
                                         const GraphSummary &summary, size_t stream_id,
                                         GeDeviceResManagerPtr res_manager) {
  AllocConstMemory(options, graph, summary.const_memory_size, res_manager);
  if (IsDisableGeKernel()) {
    AllocFeatureMemory(options, summary.fixed_memory_size, res_manager);
  } else {
    GEMemoryManager::Instance().InitGEMemory(options, summary.workspace_memory_size, summary.fixed_memory_size,
                                             summary.const_memory_size, summary.is_refreshable, stream_id);
  }
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
