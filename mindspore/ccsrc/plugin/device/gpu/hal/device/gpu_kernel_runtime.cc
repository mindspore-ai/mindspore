/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/hal/device/gpu_kernel_runtime.h"
#include <algorithm>
#include <map>
#include <chrono>
#include "include/common/debug/anf_dump_utils.h"
#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include "plugin/device/gpu/hal/device/cuda_driver.h"
#include "plugin/device/gpu/hal/device/gpu_event.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "distributed/init.h"
#include "include/common/utils/convert_utils.h"
#include "utils/ms_context.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "utils/ms_utils.h"
#include "plugin/device/gpu/hal/device/gpu_memory_manager.h"
#include "kernel/common_utils.h"
#include "plugin/device/gpu/hal/device/gpu_memory_copy_manager.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "ir/dtype.h"
#include "include/backend/optimizer/helper.h"
#ifndef ENABLE_SECURITY
#include "plugin/device/gpu/hal/profiler/gpu_profiling.h"
#include "plugin/device/gpu/hal/profiler/gpu_profiling_utils.h"
#endif
#include "utils/shape_utils.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#include "debug/rdr/mem_address_recorder.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
using mindspore::device::memswap::MemSwapInfoSet;
using mindspore::device::memswap::MemSwapManager;
using mindspore::device::memswap::SwapKind;
static const size_t PARAMETER_OUTPUT_INDEX = 0;
static thread_local bool cur_thread_device_inited{false};

bool GPUKernelRuntime::SyncStream() {
  if (!GPUDeviceManager::GetInstance().SyncStream(stream_)) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(ERROR) << "Call SyncStream error.";
    return false;
  }
  return true;
}

bool GPUKernelRuntime::Init() {
  enable_relation_cache_ = graphkernel::GraphKernelFlags::GetInstance().IsEnableGraphKernel();

  if (device_init_) {
    if (!cur_thread_device_inited) {
      CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SetDevice(UintToInt(device_id_)), "Failed to set device id");
      cur_thread_device_inited = true;
    }
    GPUMemoryAllocator::GetInstance().CheckMaxDeviceMemory();
    return true;
  }
  bool ret = InitDevice();
  if (!ret) {
    MS_LOG(ERROR) << "InitDevice error.";
    return ret;
  }
#ifndef ENABLE_SECURITY
  DumpJsonParser::GetInstance().Parse();
#endif
  mem_manager_ = std::make_shared<GPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->Initialize();
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
#if defined(_WIN32)
    MS_LOG(EXCEPTION) << "windows not support nccl.";
#endif
  }
  device_init_ = true;

#ifdef ENABLE_DEBUGGER
  SetDebugger();
#endif

  return ret;
}

namespace {
std::vector<int> CheckRealOutput(const std::string &node_name, const size_t &output_size) {
  // define a vector containing real output number
  std::vector<int> real_outputs;
  // P.BatchNorm is used for training and inference
  // can add the filter list for more operators here....
  if (node_name == "BatchNorm") {
    MS_LOG(INFO) << "loading node named " << node_name;
    real_outputs.insert(real_outputs.end(), {0, 3, 4});
  } else {
    // by default, TensorLoader will load all outputs
    for (size_t j = 0; j < output_size; ++j) {
      real_outputs.push_back(j);
    }
  }
  return real_outputs;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime.
 * Description: Load data and dump the node if needed.
 */
#ifdef ENABLE_DEBUGGER
void LoadKernelData(Debugger *debugger, const CNodePtr &kernel,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_inputs,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_workspaces,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_outputs, int exec_order, void *stream_ptr,
                    bool dump_enabled, bool last_kernel) {
  MS_EXCEPTION_IF_NULL(debugger);
  MS_EXCEPTION_IF_NULL(kernel);
  // check if we should read the kernel data
  bool read_data = false;
  auto &dump_json_parser = DumpJsonParser::GetInstance();
  std::string kernel_name = GetKernelNodeName(kernel);
  debugger->SetCurNode(kernel_name);
  if (dump_enabled) {
    auto dump_mode = dump_json_parser.dump_mode();
    // dump the node if dump_mode is 0, which means all kernels, or if this kernel is in the kernels list
    if ((dump_mode == 0) || ((dump_mode == 1) && dump_json_parser.NeedDump(kernel_name))) {
      read_data = true;
    }
  }
  if (debugger->debugger_enabled()) {
    read_data = debugger->ReadNodeDataRequired(kernel);
  }
  if (!read_data) {
    return;
  }

  if (debugger->debugger_enabled() || dump_json_parser.InputNeedDump()) {
    // get inputs
    auto input_size = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < input_size; ++j) {
      auto input_kernel = kernel->input(j + 1);
      MS_EXCEPTION_IF_NULL(input_kernel);
      std::string input_kernel_name = GetKernelNodeName(input_kernel);
      auto addr = kernel_inputs[j];
      auto type = common::AnfAlgo::GetOutputInferDataType(input_kernel, PARAMETER_OUTPUT_INDEX);
      // For example, this happens with the Depend op
      if (type == kMetaTypeNone) {
        continue;
      }
      auto format = kOpFormat_DEFAULT;
      MS_EXCEPTION_IF_NULL(addr);
      auto gpu_addr = std::make_unique<GPUDeviceAddress>(addr->addr, addr->size, format, type);
      string input_tensor_name = input_kernel_name + ':' + "0";
      ShapeVector int_shapes = trans::GetRuntimePaddingShape(input_kernel, PARAMETER_OUTPUT_INDEX);
      auto ret =
        gpu_addr->LoadMemToHost(input_tensor_name, exec_order, format, int_shapes, type, 0, true, 0, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost:"
                      << ", tensor_name:" << input_tensor_name << ", host_format:" << format << ".!";
      }
    }
  }

  if (debugger->debugger_enabled() || dump_json_parser.OutputNeedDump()) {
    // get outputs
    auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
    auto node_name = common::AnfAlgo::GetCNodeName(kernel);

    std::vector<int> real_outputs = CheckRealOutput(node_name, output_size);

    for (int j : real_outputs) {
      auto addr = kernel_outputs[j];
      auto type = common::AnfAlgo::GetOutputInferDataType(kernel, j);
      // For example, this happens with the Depend op
      if (type == kMetaTypeNone) {
        continue;
      }
      auto format = kOpFormat_DEFAULT;
      MS_EXCEPTION_IF_NULL(addr);
      auto gpu_addr = std::make_unique<GPUDeviceAddress>(addr->addr, addr->size, format, type);
      string tensor_name = kernel_name + ':' + std::to_string(j);
      ShapeVector int_shapes = trans::GetRuntimePaddingShape(kernel, j);
      auto ret = gpu_addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, j, false, 0, false, true);
      if (!ret) {
        MS_LOG(ERROR) << "LoadMemToHost:"
                      << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
      }
    }
  }
  debugger->PostExecuteNode(kernel, last_kernel);
}
#endif
}  // namespace

bool GPUKernelRuntime::MemcpyAsync(void *dst, const void *src, uint64_t size, int32_t kind) {
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  auto ret = GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(dst, src, size, stream);
  if (!ret) {
    MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
    return false;
  }
  return ret;
}

DeviceAddressPtr GPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) const {
  return std::make_shared<GPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

DeviceAddressPtr GPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id, const KernelWithIndex &node_index) const {
  return std::make_shared<GPUDeviceAddress>(device_ptr, device_size, format, type_id, node_index);
}

bool GPUKernelRuntime::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }
  if (!GPUDeviceManager::GetInstance().is_device_id_init()) {
    if (!GPUDeviceManager::GetInstance().set_cur_device_id(device_id_)) {
      MS_LOG(ERROR) << "Failed to set current device to " << SizeToInt(device_id_);
      return false;
    }
  }
  GPUDeviceManager::GetInstance().InitDevice();
  stream_ = GPUDeviceManager::GetInstance().default_stream();
  if (stream_ == nullptr) {
    MS_LOG(ERROR) << "No default CUDA stream found.";
    return false;
  }
  GPUDeviceManager::GetInstance().CreateStream(&communication_stream_);
  if (communication_stream_ == nullptr) {
    MS_LOG(ERROR) << "Invalid communication stream";
    return false;
  }
  return true;
}

void GPUKernelRuntime::ReleaseDeviceRes() {
  // For dataset mode.
#ifdef ENABLE_DEBUGGER
  if (debugger_ && debugger_->debugger_enabled()) {
    debugger_->SetTrainingDone(true);
    bool ret = debugger_->SendMetadata(false);
    if (!ret) {
      MS_LOG(ERROR) << "Failed to SendMetadata when finalize";
    }
  }
#endif
  if (DataQueueMgr::GetInstance().IsInit()) {
    if (!DataQueueMgr::GetInstance().IsClosed()) {
      if (!DataQueueMgr::GetInstance().CloseNotify()) {
        MS_LOG(ERROR) << "Could not close gpu data queue.";
      }
    }
    DataQueueMgr::GetInstance().Release();
  }

  // Destroy remaining memory swap events and free host memory.
  for (auto &item : mem_swap_map_) {
    auto &mem_swap_manager = item.second;
    MS_EXCEPTION_IF_NULL(mem_swap_manager);
    if (mem_swap_manager->trigger_swap()) {
      mem_swap_manager->ClearSwapQueue(false);
      mem_swap_manager->ReleaseHostPinnedMem();
    }
  }

  GPUDeviceManager::GetInstance().ReleaseDevice();
  if (mem_manager_ != nullptr) {
    mem_manager_->Finalize();
  }
}

void GPUKernelRuntime::ClearGraphRuntimeResource(uint32_t graph_id) {
  MS_LOG(INFO) << "Clear graph:" << graph_id << " GPU runtime resource";
  graph_output_map_.erase(graph_id);
}

void GPUKernelRuntime::AllocInplaceNodeMemory(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_alloc_inplace_res_[graph->graph_id()]) {
    return;
  }
  is_alloc_inplace_res_[graph->graph_id()] = true;

  std::map<uint32_t, std::vector<CNodePtr>> inplace_groups;
  auto kernel_cnodes = graph->execution_order();
  for (auto &kernel : kernel_cnodes) {
    if (!common::AnfAlgo::IsInplaceNode(kernel, "inplace_algo")) {
      continue;
    }
    auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
    MS_EXCEPTION_IF_NULL(primitive);
    auto group_attr = primitive->GetAttr("inplace_group");
    MS_EXCEPTION_IF_NULL(group_attr);
    auto group_id = GetValue<uint32_t>(group_attr);
    inplace_groups[group_id].push_back(kernel);
  }

  for (auto &group : inplace_groups) {
    auto &item = group.second;
    // in-place compute when group size >= 2.
    if (item.size() < 2) {
      continue;
    }

    auto primitive = common::AnfAlgo::GetCNodePrimitive(item[0]);
    MS_EXCEPTION_IF_NULL(primitive);
    auto output_index = GetValue<uint32_t>(primitive->GetAttr("inplace_output_index"));
    auto device_address = GetMutableOutputAddr(item[0], output_index, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->GetPtr() != nullptr) {
      continue;
    }

    auto kernel_mod = AnfAlgo::GetKernelMod(item[0]);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_size = kernel_mod->GetOutputSizeList();
    MS_EXCEPTION_IF_NULL(mem_manager_);
    auto ret = mem_manager_->MallocMemFromMemPool(device_address, output_size[output_index]);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Device memory isn't enough and alloc failed, alloc size:" << output_size[output_index];
    }

    for (auto &node : item) {
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      auto index = GetValue<uint32_t>(prim->GetAttr("inplace_output_index"));
      AnfAlgo::SetOutputAddr(device_address, index, node.get());
    }
  }
}

bool GPUKernelRuntime::IsDistributedTraining(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernels = graph->execution_order();
  return std::any_of(kernels.begin(), kernels.end(),
                     [](const AnfNodePtr &kernel) { return common::AnfAlgo::IsCommunicationOp(kernel); });
}

void GPUKernelRuntime::FetchMemUnitSize(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  mem_reuse_util_->ResetDynamicUsedRefCount();
  size_t max_sum_size = 0;
  size_t current_sum_size = 0;
  constexpr size_t kZeroNumber = 0;
  auto &kernels = graph->execution_order();
  for (const auto &cnode : kernels) {
    auto kernel_mode = AnfAlgo::GetKernelMod(cnode);
    MS_EXCEPTION_IF_NULL(kernel_mode);
    auto kernel = cnode->cast<AnfNodePtr>();
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsCommunicationOp(kernel)) {
      continue;
    }
    const auto &input_size_list = kernel_mode->GetInputSizeList();
    const auto &output_size_list = kernel_mode->GetOutputSizeList();
    const auto &workspace_size_list = kernel_mode->GetWorkspaceSizeList();

    size_t input_size = std::accumulate(input_size_list.begin(), input_size_list.end(), kZeroNumber);
    size_t output_size = std::accumulate(output_size_list.begin(), output_size_list.end(), kZeroNumber);
    size_t workspace_size = std::accumulate(workspace_size_list.begin(), workspace_size_list.end(), kZeroNumber);
    current_sum_size = current_sum_size + input_size + output_size + workspace_size;
    if (current_sum_size > max_sum_size) {
      max_sum_size = current_sum_size;
    }

    // Free the input of kernel by reference count.
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    if (input_num != input_size_list.size()) {
      continue;
    }
    for (size_t i = 0; i < input_num; ++i) {
      auto kernel_ref_count_ptr = mem_reuse_util_->GetKernelInputRef(cnode, i);
      if (kernel_ref_count_ptr == nullptr) {
        continue;
      }
      kernel_ref_count_ptr->ref_count_dynamic_use_--;
      if (kernel_ref_count_ptr->ref_count_dynamic_use_ < 0) {
        MS_LOG(EXCEPTION) << "Check dynamic reference count failed.";
      }
      if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
        auto remove_size = kernel_ref_count_ptr->ref_count_ * input_size_list.at(i);
        if (remove_size <= current_sum_size) {
          current_sum_size -= remove_size;
        } else {
          current_sum_size = 0;
        }
      }
    }
    auto output_workspace_size = output_size + workspace_size;
    if (output_workspace_size <= current_sum_size) {
      current_sum_size -= output_workspace_size;
    } else {
      current_sum_size = 0;
    }
  }
  if (max_sum_size > GPUMemoryAllocator::GetInstance().MemAllocUnitSize()) {
    size_t unit_size = (max_sum_size / DYNAMIC_MEM_ALLOC_UNIT_SIZE + 1) * DYNAMIC_MEM_ALLOC_UNIT_SIZE;
    if (unit_size < DYNAMIC_MEM_ALLOC_UNIT_SIZE) {
      MS_LOG(WARNING) << "Current memory unit size [" << unit_size << "] is too small.";
      return;
    }
    size_t free_mem_size = GPUMemoryAllocator::GetInstance().free_mem_size();
    constexpr float kValidMemoryRatio = 0.9;
    free_mem_size = kValidMemoryRatio * free_mem_size;
    unit_size = std::min(unit_size, free_mem_size);
    GPUMemoryAllocator::GetInstance().SetMemAllocUintSize(unit_size);
  }
}

void GPUKernelRuntime::AssignMemory(const session::KernelGraph &graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  bool is_enable_dynamic_mem = context_ptr->get_param<bool>(MS_CTX_ENABLE_DYNAMIC_MEM_POOL);
  if (is_enable_dynamic_mem) {
    // Use the dynamic memory pool.
    InitKernelRefCount(&graph);
    InitMemorySwapInfo(&graph);
    InitKernelOutputAddress(&graph);
    InitKernelWorkspaceAddress(&graph);
    SaveGraphOutputNode(&graph);
  } else {
    AssignDynamicMemory(graph);
  }
}

bool GPUKernelRuntime::Run(const session::KernelGraph &graph, bool is_task_sink) {
  std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
  bool ret = true;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_dynamic_mem = context_ptr->get_param<bool>(MS_CTX_ENABLE_DYNAMIC_MEM_POOL);
  bool is_enable_pynative_infer = context_ptr->get_param<bool>(MS_CTX_ENABLE_PYNATIVE_INFER);
  bool is_pynative_mode = (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode);
  if (is_enable_dynamic_mem && !is_pynative_mode && !is_enable_pynative_infer) {
    auto graph_id = graph.graph_id();
    auto iter = mem_swap_map_.find(graph_id);
    if (iter == mem_swap_map_.end()) {
      MS_LOG(EXCEPTION) << "Find memory swap map failed.";
    }
    mem_swap_manager_ = iter->second;
    MS_EXCEPTION_IF_NULL(mem_swap_manager_);
    auto mem_reuse_iter = mem_reuse_util_map_.find(graph_id);
    if (mem_reuse_iter == mem_reuse_util_map_.end()) {
      MS_LOG(EXCEPTION) << "Find memory reuse map failed.";
    }
    mem_reuse_util_ = mem_reuse_iter->second;
    MS_EXCEPTION_IF_NULL(mem_reuse_util_);

    ret = RunOneStep(&graph);
  } else {
    if (graph.is_dynamic_shape()) {
      // run dynamic shape graph in pynative
      ret = RunOpLaunchKernelDynamic(&graph);
    } else {
      ret = LaunchKernels(graph);
    }
  }
  std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
  auto ms_duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
  uint64_t cost = ms_duration.count();
  MS_LOG(DEBUG) << "GPU kernel runtime run graph in " << cost << " us";
  return ret;
}

std::shared_ptr<DeviceEvent> GPUKernelRuntime::CreateDeviceEvent() {
  auto gpu_event = std::make_shared<GpuEvent>();
  MS_EXCEPTION_IF_NULL(gpu_event);
  return gpu_event;
}

bool GPUKernelRuntime::RunOneStep(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_id = graph->graph_id();
  if (!is_first_step_map_[graph_id] || graph->is_dynamic_shape()) {
    // Normally run graph
    return LaunchKernelDynamic(graph);
  }
  // Mock run first step
  FetchMemUnitSize(graph);
  bool ret = LaunchKernelDynamic(graph, true, false);
  is_first_step_map_[graph_id] = false;
  if (ret) {
    // Normally run graph
    return LaunchKernelDynamic(graph);
  }
  if (IsDistributedTraining(graph)) {
    MS_LOG(ERROR) << "Device memory is not enough, run graph failed!";
    return false;
  }
  // Trigger memory swap
  return SearchMemSwapScheme(graph);
}

bool GPUKernelRuntime::SearchMemSwapScheme(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  MS_LOG(INFO) << "Run out of memory and try memory swapping, it may take some time, please wait a moment.";
  bool ret = false;
  ClearKernelOldOutputAndWorkspace(graph);
  if (!mem_swap_manager_->mem_swap_init()) {
    if (!mem_swap_manager_->Init(graph)) {
      return false;
    }
  }

  while (!ret) {
    if (!mem_swap_manager_->RetreatSwapInfo()) {
      MS_LOG(ERROR) << "Device memory is not enough, run graph failed!";
      return false;
    }
    ret = LaunchKernelDynamic(graph, true, false);
    if (!ret) {
      ClearKernelOldOutputAndWorkspace(graph);
    }
  }
  mem_swap_manager_->AssignHostMemory();

  // Time profiling
  ret = LaunchKernelDynamic(graph, false, true);
  if (!ret) {
    return ret;
  }
  return RefineMemSwapScheme(graph);
}

bool GPUKernelRuntime::RefineMemSwapScheme(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  MS_LOG(INFO) << "Refine memory swap scheme, it may take some time, please wait a moment.";
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    if (!mem_swap_manager_->QueryKernelTriggerSwapIn(kernel)) {
      continue;
    }

    size_t swap_in_task_num = mem_swap_manager_->QueryKernelTriggerSwapInTaskNum(kernel);
    for (size_t swap_in_task_idx = 0; swap_in_task_idx < swap_in_task_num; swap_in_task_idx++) {
      bool ret = false;
      while (!ret) {
        mem_swap_manager_->AdjustSwapInPos(kernel, swap_in_task_idx);
        ret = LaunchKernelDynamic(graph, true, false);
        if (!ret) {
          ClearKernelOldOutputAndWorkspace(graph);
          ClearSwapInfo(true);
        }
      }
    }
  }
  return true;
}

void GPUKernelRuntime::InitKernelRefCount(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MemReuseUtilPtr mem_reuse_util_ptr = std::make_shared<memreuse::MemReuseUtil>();
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  // Init the kernel reference count.
  if (!mem_reuse_util_ptr->InitDynamicKernelRef(graph)) {
    MS_LOG(EXCEPTION) << "Init kernel reference count failed";
  }
  mem_reuse_util_ptr->SetKernelDefMap();
  mem_reuse_util_ptr->SetReuseRefCount();
  // Can't free the device address of graph output, so set the reference count of graph output specially.
  mem_reuse_util_ptr->SetGraphOutputRefCount();
#ifndef ENABLE_SECURITY
  // Can't free the device address of summary nodes, so set the reference count of summary nodes specially.
  mem_reuse_util_ptr->SetSummaryNodesRefCount();
#endif
  auto graph_id = graph->graph_id();
  mem_reuse_util_map_[graph_id] = mem_reuse_util_ptr;
}

void GPUKernelRuntime::InitMemorySwapInfo(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  GPUMemCopyManagerPtr gpu_mem_copy_manager = std::make_shared<GPUMemCopyManager>();
  MS_EXCEPTION_IF_NULL(gpu_mem_copy_manager);
  MemSwapManagerPtr mem_swap_manager = std::make_shared<MemSwapManager>(gpu_mem_copy_manager);
  MS_EXCEPTION_IF_NULL(mem_swap_manager);
  auto graph_id = graph->graph_id();
  mem_swap_map_[graph_id] = mem_swap_manager;
  is_first_step_map_[graph_id] = true;
}

void GPUKernelRuntime::InitKernelOutputAddress(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      if (AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }
      std::string output_format = AnfAlgo::GetOutputFormat(kernel, i);
      auto output_type = AnfAlgo::GetOutputDeviceDataType(kernel, i);
      auto device_address = CreateDeviceAddress(nullptr, output_sizes[i], output_format, output_type);
      AnfAlgo::SetOutputAddr(device_address, i, kernel.get());
    }
  }
}

void GPUKernelRuntime::InitKernelWorkspaceAddress(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      auto device_address = CreateDeviceAddress(nullptr, workspace_sizes[i], "", kTypeUnknown);
      AnfAlgo::SetWorkspaceAddr(device_address, i, kernel.get());
    }
  }
}

void GPUKernelRuntime::SaveGraphOutputNode(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_id = graph->graph_id();
  const auto &output_nodes = common::AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
  for (const auto &node : output_nodes) {
    graph_output_map_[graph_id].insert(node);
  }
}

bool GPUKernelRuntime::IsGraphOutput(const session::KernelGraph *graph, const mindspore::AnfNodePtr &kernel) const {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_id = graph->graph_id();
  auto iter = graph_output_map_.find(graph_id);
  if (iter == graph_output_map_.end()) {
    MS_LOG(EXCEPTION) << "Find graph output info failed.";
  }
  auto &graph_output_set = iter->second;
  return (graph_output_set.find(kernel) != graph_output_set.end());
}

void GPUKernelRuntime::ClearKernelOldOutputAndWorkspace(const session::KernelGraph *graph) {
  ClearKernelOutputAddress(graph);
  ClearKernelWorkspaceAddress(graph);
}

void GPUKernelRuntime::ClearKernelOutputAddress(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    if (IsGraphOutput(graph, kernel)) {
      continue;
    }
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      if (!AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->ptr_) {
        mem_manager_->FreeMemFromMemPool(device_address);
      }
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
}

void GPUKernelRuntime::ClearKernelWorkspaceAddress(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto workspace_sizes = kernel_mod->GetWorkspaceSizeList();
    for (size_t i = 0; i < workspace_sizes.size(); ++i) {
      auto device_address = AnfAlgo::GetMutableWorkspaceAddr(kernel, i);
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->ptr_) {
        mem_manager_->FreeMemFromMemPool(device_address);
      }
    }
  }
}

CNodePtr GetLastKernel(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernels = graph->execution_order();
  CNodePtr last_kernel;
  for (const auto &kernel : kernels) {
    if (common::AnfAlgo::IsInplaceNode(kernel, "skip")) {
      continue;
    } else {
      last_kernel = kernel;
    }
  }
  return last_kernel;
}

bool GPUKernelRuntime::LaunchKernelDynamic(const session::KernelGraph *graph, bool mock, bool profiling) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_reuse_util_);
  // Reset the reference count.
  mem_reuse_util_->ResetDynamicUsedRefCount();
  // The inputs and outputs memory of communication kernel need be continuous, so separate processing.
  AllocCommunicationOpDynamicRes(graph);
  AllocInplaceNodeMemory(graph);

#ifdef ENABLE_DEBUGGER
  bool dump_enabled = GPUKernelRuntime::DumpDataEnabledIteration();
  if (!mock && debugger_) {
    // Update the step number for old GPU runtime.
    debugger_->UpdateStepNum(graph);
  }
#endif
  auto &kernels = graph->execution_order();
  int exec_order = 1;
#ifdef ENABLE_DUMP_IR
  std::string name = "mem_address_list";
  (void)mindspore::RDR::RecordMemAddressInfo(SubModuleId::SM_KERNEL, name);
#endif
  CNodePtr last_kernel = GetLastKernel(graph);
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    if (common::AnfAlgo::IsInplaceNode(kernel, "skip")) {
      continue;
    }

    // akg kernel do not support dynamic shape by now.
    kernel::NativeGpuKernelMod *gpu_kernel = nullptr;
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) != KernelType::AKG_KERNEL) {
      gpu_kernel = dynamic_cast<kernel::NativeGpuKernelMod *>(kernel_mod);
      MS_EXCEPTION_IF_NULL(gpu_kernel);
    }

    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      opt::InferOp(kernel);
      auto args = kernel::GetArgsFromCNode(kernel);
      if (gpu_kernel->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
          kernel::KRET_RESIZE_FAILED) {
        MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed.";
      }
    }

    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    auto ret = AllocKernelDynamicRes(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs, mock);
    if (!ret) {
#ifdef ENABLE_DEBUGGER
      if (!mock) {
        MS_EXCEPTION_IF_NULL(debugger_);
        // invalidate current data collected by the debugger
        debugger_->ClearCurrentData();
      }
#endif
      return false;
    }
#ifdef ENABLE_DUMP_IR
    kernel::KernelLaunchInfo mem_info = {kernel_inputs, kernel_workspaces, kernel_outputs};
    std::string op_name = kernel->fullname_with_scope();
    (void)mindspore::RDR::UpdateMemAddress(SubModuleId::SM_KERNEL, name, op_name, mem_info);
#endif
    if (!mock) {
      LaunchKernelWithoutMock(graph, kernel, kernel_inputs, kernel_workspaces, kernel_outputs, profiling);

      if (gpu_kernel != nullptr && common::AnfAlgo::IsDynamicShape(kernel)) {
        kernel::UpdateNodeShape(kernel);
      }
#ifdef ENABLE_DEBUGGER
      MS_EXCEPTION_IF_NULL(debugger_);
      // called once per kernel to collect the outputs to the kernel (does a SyncDeviceToHost)
      LoadKernelData(debugger_.get(), kernel, kernel_inputs, kernel_workspaces, kernel_outputs, exec_order, stream_,
                     dump_enabled, kernel == last_kernel);
#endif
    }
    exec_order = exec_order + 1;
    FreeKernelDynamicRes(kernel);
    if (!UpdateMemorySwapTask(kernel, mock, profiling)) {
#ifdef ENABLE_DEBUGGER
      if (!mock) {
        MS_EXCEPTION_IF_NULL(debugger_);
        // invalidate current data collected by the debugger
        debugger_->ClearCurrentData();
      }
#endif
      return false;
    }
  }
  if (!mock) {
    auto context_ptr = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context_ptr);
    if (context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE) != kPynativeMode) {
      CHECK_OP_RET_WITH_EXCEPT(SyncStream(), "SyncStream failed.");
    }
  }
  ClearSwapInfo(mock);
  return true;
}

void *GPUKernelRuntime::GetKernelStream(const AnfNodePtr &kernel) const {
  auto stream = GPUDeviceManager::GetInstance().GetStream(AnfAlgo::GetStreamId(kernel));
  if (stream == nullptr) {
    return GPUDeviceManager::GetInstance().default_stream();
  }
  return stream;
}

void GPUKernelRuntime::LaunchKernelWithoutMock(const session::KernelGraph *graph, const AnfNodePtr &kernel,
                                               const AddressPtrList &inputs, const AddressPtrList &workspaces,
                                               const AddressPtrList &outputs, bool profiling) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(kernel);
#ifndef ENABLE_SECURITY
  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);

  if (profiler_inst->GetEnableFlag() && profiler::gpu::ProfilingUtils::IsFirstStep(graph->graph_id())) {
    profiler::gpu::ProfilingTraceInfo profiling_trace =
      profiler::gpu::ProfilingUtils::GetProfilingTraceFromEnv(NOT_NULL(graph));
    profiler_inst->SetStepTraceOpName(profiling_trace);
  }
#endif
  if (!profiling) {
#ifndef ENABLE_SECURITY
    if (profiler_inst->GetEnableFlag()) {
      profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), stream_);
    }
#endif
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    MS_EXCEPTION_IF_NULL(stream_);
    if (!kernel_mod->Launch(inputs, workspaces, outputs, stream_)) {
#ifdef ENABLE_DUMP_IR
      mindspore::RDR::TriggerAll();
#endif
      MS_LOG(EXCEPTION) << "Launch kernel failed: " << kernel->fullname_with_scope();
    }
#ifndef ENABLE_SECURITY
    if (profiler_inst->GetEnableFlag()) {
      profiler_inst->OpDataProducerEnd();
      if (profiler_inst->GetSyncEnableFlag()) {
        CHECK_OP_RET_WITH_ERROR(SyncStream(), "Profiler SyncStream failed.");
      }
    }
#endif
  } else {
    LaunchKernelWithTimeProfiling(kernel, inputs, workspaces, outputs);
  }
}

bool GPUKernelRuntime::RunOpLaunchKernelDynamic(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  const auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    // akg kernel do not support dynamic shape by now.
    kernel::NativeGpuKernelMod *gpu_kernel = nullptr;
    if (session::AnfRuntimeAlgorithm::GetKernelType(kernel) != KernelType::AKG_KERNEL) {
      gpu_kernel = dynamic_cast<kernel::NativeGpuKernelMod *>(kernel_mod);
      MS_EXCEPTION_IF_NULL(gpu_kernel);
    }
    // pre-processing for dynamic shape kernel
    if (common::AnfAlgo::IsDynamicShape(kernel)) {
      opt::InferOp(kernel);
      auto args = kernel::GetArgsFromCNode(kernel);
      if (gpu_kernel->Resize(args->op, args->inputs, args->outputs, args->depend_tensor_map) ==
          kernel::KRET_RESIZE_FAILED) {
        MS_LOG(EXCEPTION) << "Node " << kernel->fullname_with_scope() << " Resize failed.";
      }
    }
    // alloc kernel res
    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    KernelLaunchInfo kernel_launch_info;
    GenLaunchArgs(*kernel_mod, kernel, &kernel_launch_info);
    MS_EXCEPTION_IF_NULL(stream_);
    auto ret = kernel_mod->Launch(kernel_launch_info, stream_);
    if (!ret) {
      MS_LOG(ERROR) << "Launch kernel failed.";
      return false;
    }
    if (gpu_kernel != nullptr && common::AnfAlgo::IsDynamicShape(kernel)) {
      kernel::UpdateNodeShape(kernel);
    }
  }
  return true;
}

void GPUKernelRuntime::LaunchKernelWithTimeProfiling(const AnfNodePtr &kernel, const AddressPtrList &inputs,
                                                     const AddressPtrList &workspace, const AddressPtrList &outputs) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  float cost_time = 0;
  CudaDeviceStream start = nullptr;
  CudaDeviceStream end = nullptr;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&start), "Failed to create event.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ConstructEvent(&end), "Failed to create event.");

  MS_EXCEPTION_IF_NULL(stream_);
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(start, stream_), "Failed to record event to stream.");
  CHECK_OP_RET_WITH_EXCEPT(kernel_mod->Launch(inputs, workspace, outputs, stream_), "Launch kernel failed.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::RecordEvent(end, stream_), "Failed to record event to stream.");

  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SyncEvent(start), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::SyncEvent(end), "Failed to sync event.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::ElapsedTime(&cost_time, start, end), "Failed to record elapsed time.");

  mem_swap_manager_->AddKernelExecutionPerform(kernel, cost_time);

  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(start), "Failed to destroy event.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::DestroyEvent(end), "Failed to destroy event.");
}

bool GPUKernelRuntime::AddMemorySwapTask(const AnfNodePtr &kernel, bool mock, bool profiling) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  const MemSwapInfoSet &mem_swap_info_set = mem_swap_manager_->QueryKernelMemSwapInfo(kernel);
  for (auto &mem_swap_info : mem_swap_info_set) {
    auto need_swap_kernel = mem_swap_manager_->QueryKernelByTopoOrder(mem_swap_info.topo_order_);
    MS_EXCEPTION_IF_NULL(need_swap_kernel);
    const HostAddress &host_address =
      mem_swap_manager_->QueryKernelHostAddr(need_swap_kernel, mem_swap_info.output_idx_);
    auto device_address = GetMutableOutputAddr(need_swap_kernel, mem_swap_info.output_idx_, false);
    MS_EXCEPTION_IF_NULL(device_address);

    if (mem_swap_info.swap_kind_ == SwapKind::kDeviceToHost) {
      if (mem_swap_manager_->QueryKernelHostAddrIsDirty(need_swap_kernel, mem_swap_info.output_idx_)) {
        mem_swap_manager_->AddMemSwapTask(SwapKind::kDeviceToHost, device_address, host_address, mock);
        mem_swap_manager_->AddKernelHostAddrIsDirty(need_swap_kernel, mem_swap_info.output_idx_, false);
      } else {
        mem_manager_->FreeMemFromMemPool(device_address);
        device_address->set_status(DeviceAddressStatus::kInHost);
      }
    } else if (mem_swap_info.swap_kind_ == SwapKind::kHostToDevice) {
      auto status = device_address->status();
      if (status == DeviceAddressStatus::kInDeviceToHost) {
        device_address->set_status(DeviceAddressStatus::kInDevice);
      } else if (status == DeviceAddressStatus::kInHost) {
        if (!device_address->ptr_ && !AttemptMallocMem(device_address, device_address->size_, mock)) {
          return false;
        }
        float cost_time = 0;
        mem_swap_manager_->AddMemSwapTask(SwapKind::kHostToDevice, device_address, host_address, mock, profiling,
                                          &cost_time);
        if (profiling) {
          mem_swap_manager_->AddKernelSwapPerform(need_swap_kernel, mem_swap_info.output_idx_,
                                                  std::make_pair(0, cost_time));
        }
      }
    }
  }
  return true;
}

bool GPUKernelRuntime::UpdateMemorySwapTask(const AnfNodePtr &kernel, bool mock, bool profiling) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  if (!mem_swap_manager_->trigger_swap()) {
    return true;
  }
  if (mem_swap_manager_->QueryKernelTriggerSwap(kernel)) {
    if (!mock) {
      CHECK_OP_RET_WITH_EXCEPT(SyncStream(), "SyncStream failed.");
    }
    if (!AddMemorySwapTask(kernel, mock, profiling)) {
      return false;
    }
    if (!mock) {
      CHECK_OP_RET_WITH_EXCEPT(mem_swap_manager_->SyncMemCopyStream(SwapKind::kDeviceToHost), "SyncCopyStream failed.");
    }
  }
  return true;
}

void GPUKernelRuntime::UpdateHostSwapInQueue(const DeviceAddressPtr device_address, bool mock) {
  MS_EXCEPTION_IF_NULL(device_address);
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  if (!mem_swap_manager_->trigger_swap()) {
    return;
  }
  while (auto device_address_swap_in = mem_swap_manager_->UpdateSwapQueue(SwapKind::kHostToDevice, mock)) {
    device_address_swap_in->set_status(DeviceAddressStatus::kInDevice);
  }

  auto status = device_address->status();
  switch (status) {
    case DeviceAddressStatus::kInDevice:
      break;
    case DeviceAddressStatus::kInDeviceToHost: {
      device_address->set_status(DeviceAddressStatus::kInDevice);
      break;
    }
    case DeviceAddressStatus::kInHostToDevice: {
      while (device_address->status() != DeviceAddressStatus::kInDevice) {
        while (auto device_address_swap_in = mem_swap_manager_->UpdateSwapQueue(SwapKind::kHostToDevice, mock)) {
          device_address_swap_in->set_status(DeviceAddressStatus::kInDevice);
        }
      }
      break;
    }
    case DeviceAddressStatus::kInHost:
      MS_LOG(WARNING) << "Unexpected device address status: " << status;
      break;
    default:
      MS_LOG(EXCEPTION) << "Invalid device address status: " << status;
  }
}

void GPUKernelRuntime::UpdateHostSwapOutQueue(bool mock) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (!mem_swap_manager_->trigger_swap()) {
    return;
  }
  while (auto device_address_swap_out = mem_swap_manager_->UpdateSwapQueue(SwapKind::kDeviceToHost, mock)) {
    if (device_address_swap_out->status() == DeviceAddressStatus::kInDeviceToHost && device_address_swap_out->ptr_) {
      device_address_swap_out->set_status(DeviceAddressStatus::kInHost);
      mem_manager_->FreeMemFromMemPool(device_address_swap_out);
    }
  }
}

void GPUKernelRuntime::ClearSwapInfo(bool mock) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  if (!mem_swap_manager_->trigger_swap()) {
    return;
  }
  mem_swap_manager_->ClearSwapQueue(mock);
  mem_swap_manager_->ResetHostAddrIsDirty();
}

bool GPUKernelRuntime::AttemptMallocMem(const DeviceAddressPtr &device_address, size_t size, bool mock) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  auto ret = mem_manager_->MallocMemFromMemPool(device_address, size);
  if (!ret) {
    if (!mem_swap_manager_->trigger_swap()) {
      return false;
    }
    if (!mock) {
      mem_swap_manager_->SyncMemCopyStream(SwapKind::kDeviceToHost);
    }
    UpdateHostSwapOutQueue(mock);

    ret = mem_manager_->MallocMemFromMemPool(device_address, size);
    if (!ret) {
      if (!mock) {
        auto context_ptr = MsContext::GetInstance();
        MS_EXCEPTION_IF_NULL(context_ptr);
        auto device_id = context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID);
        MS_LOG(EXCEPTION) << "Device(id:" << device_id << ") memory isn't enough and alloc failed, alloc size:" << size;
      }
      return false;
    }
  }
  return true;
}

bool GPUKernelRuntime::AllocKernelDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                             const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs,
                                             AddressPtrList *kernel_workspaces, AddressPtrList *kernel_outputs,
                                             bool mock) {
  if (!AllocKernelInputDynamicRes(kernel, kernel_inputs, mock)) {
    return false;
  }
  if (!AllocKernelOutputDynamicRes(kernel_mod, kernel, kernel_outputs, mock)) {
    return false;
  }
  if (!AllocKernelWorkspaceDynamicRes(kernel_mod, kernel, kernel_workspaces, mock)) {
    return false;
  }
  return true;
}

bool GPUKernelRuntime::AllocKernelInputDynamicRes(const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs,
                                                  bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(mem_reuse_util_);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_num; ++i) {
    DeviceAddressPtr device_address;
    if (mem_reuse_util_->is_all_nop_node()) {
      // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
      device_address = GetPrevNodeMutableOutputAddr(kernel, i, false);
    } else {
      // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
      device_address = GetPrevNodeMutableOutputAddr(kernel, i, true);
    }

    // Get in-place output_address
    if (common::AnfAlgo::IsInplaceNode(kernel, "aggregate")) {
      auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
      MS_EXCEPTION_IF_NULL(primitive);
      auto input_index = GetValue<uint32_t>(primitive->GetAttr("aggregate_input_index"));
      if (i == input_index) {
        auto skip_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(kernel), input_index);
        device_address = GetPrevNodeMutableOutputAddr(skip_node, 0, false);
      }
    }

    MS_EXCEPTION_IF_NULL(device_address);
    UpdateHostSwapInQueue(device_address, mock);
    MS_EXCEPTION_IF_NULL(device_address->ptr_);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    input->size = device_address->size_;
    kernel_inputs->emplace_back(input);
  }
  return true;
}

bool GPUKernelRuntime::AllocKernelOutputDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                                   const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_outputs,
                                                   bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  UpdateHostSwapOutQueue(mock);
  if (common::AnfAlgo::IsCommunicationOp(kernel)) {
    AllocCommunicationOpOutputDynamicRes(kernel);
  }
  auto output_sizes = kernel_mod.GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr && !AttemptMallocMem(device_address, output_sizes[i], mock)) {
      return false;
    }
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = device_address->ptr_;
    output->size = output_sizes[i];
    kernel_outputs->emplace_back(output);
  }
  return true;
}

bool GPUKernelRuntime::AllocKernelWorkspaceDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                                      const mindspore::AnfNodePtr &kernel,
                                                      AddressPtrList *kernel_workspaces, bool mock) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  auto workspace_sizes = kernel_mod.GetWorkspaceSizeList();
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    if (workspace_sizes[i] == 0) {
      kernel_workspaces->emplace_back(nullptr);
      continue;
    }
    auto device_address = AnfAlgo::GetMutableWorkspaceAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr && !AttemptMallocMem(device_address, workspace_sizes[i], mock)) {
      return false;
    }
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_address->ptr_;
    workspace->size = workspace_sizes[i];
    kernel_workspaces->emplace_back(workspace);
  }
  return true;
}

void GPUKernelRuntime::AllocCommunicationOpDynamicRes(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  if (is_alloc_communication_res_[graph->graph_id()]) {
    return;
  }
  is_alloc_communication_res_[graph->graph_id()] = true;

  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (common::AnfAlgo::IsCommunicationOp(kernel) && common::AnfAlgo::GetCNodeName(kernel) != kHcomSendOpName &&
        common::AnfAlgo::GetCNodeName(kernel) != kReceiveOpName) {
      AllocCommunicationOpInputDynamicRes(kernel);
    }
  }
}

void GPUKernelRuntime::AllocCommunicationOpInputDynamicRes(const mindspore::AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_reuse_util_);
  bool is_need_alloc_memory = false;
  bool is_need_free_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  DeviceAddressPtrList addr_list;
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto intput_sizes = kernel_mod->GetInputSizeList();
  for (size_t i = 0; i < intput_sizes.size(); ++i) {
    DeviceAddressPtr device_address;
    if (mem_reuse_util_->is_all_nop_node()) {
      // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
      device_address = GetPrevNodeMutableOutputAddr(kernel, i, false);
    } else {
      // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
      device_address = GetPrevNodeMutableOutputAddr(kernel, i, true);
    }
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr) {
      is_need_alloc_memory = true;
    } else {
      is_need_free_memory = true;
    }
    total_size += intput_sizes[i];
    size_list.emplace_back(intput_sizes[i]);
    addr_list.emplace_back(device_address);
  }
  AllocCommunicationOpMemory(is_need_alloc_memory, is_need_free_memory, addr_list, total_size, size_list);
}

void GPUKernelRuntime::AllocCommunicationOpOutputDynamicRes(const mindspore::AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  bool is_need_alloc_memory = false;
  bool is_need_free_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  DeviceAddressPtrList addr_list;
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = GetMutableOutputAddr(kernel, i, false);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr) {
      is_need_alloc_memory = true;
    } else {
      is_need_free_memory = true;
    }
    total_size += output_sizes[i];
    size_list.emplace_back(output_sizes[i]);
    addr_list.emplace_back(device_address);
  }
  AllocCommunicationOpMemory(is_need_alloc_memory, is_need_free_memory, addr_list, total_size, size_list);
}

void GPUKernelRuntime::AllocCommunicationOpMemory(bool is_need_alloc_memory, bool, const DeviceAddressPtrList addr_list,
                                                  size_t total_size, std::vector<size_t> size_list) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto ret = mem_manager_->MallocContinuousMemFromMemPool(addr_list, total_size, size_list);
  if (!ret) {
    MS_LOG(EXCEPTION) << "Malloc device memory failed.";
  }
}

void GPUKernelRuntime::FreeKernelDynamicRes(const mindspore::AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  MS_EXCEPTION_IF_NULL(mem_reuse_util_);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Can not free the input addr of communication op when enable multi stream
  if (common::AnfAlgo::IsCommunicationOp(kernel)) {
    return;
  }
  // Free the input of kernel by reference count.
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel);
  for (size_t i = 0; i < input_num; ++i) {
    if (common::AnfAlgo::IsInplaceNode(kernel, "aggregate")) {
      auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel);
      MS_EXCEPTION_IF_NULL(primitive);
      auto index = GetValue<uint32_t>(primitive->GetAttr("aggregate_input_index"));
      if (i == index) {
        continue;
      }
    }

    auto kernel_ref_count_ptr = mem_reuse_util_->GetKernelInputRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    kernel_ref_count_ptr->ref_count_dynamic_use_--;
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ < 0) {
      MS_LOG(EXCEPTION) << "Check dynamic reference count failed.";
    }
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      DeviceAddressPtr device_address;
      if (mem_reuse_util_->is_all_nop_node()) {
        // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
        device_address = GetPrevNodeMutableOutputAddr(kernel, i, false);
      } else {
        // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
        device_address = GetPrevNodeMutableOutputAddr(kernel, i, true);
      }
      mem_manager_->FreeMemFromMemPool(device_address);
      MS_EXCEPTION_IF_NULL(device_address);
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
  // Free the output of kernel, if output has no reference.
  size_t output_num = AnfAlgo::GetOutputTensorNum(kernel);
  for (size_t i = 0; i < output_num; ++i) {
    auto kernel_ref_count_ptr = mem_reuse_util_->GetRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      auto device_address = GetMutableOutputAddr(kernel, i, false);
      MS_EXCEPTION_IF_NULL(device_address);
      mem_manager_->FreeMemFromMemPool(device_address);
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
  // Free the workspace of kernel.
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  for (size_t i = 0; i < kernel_mod->GetWorkspaceSizeList().size(); ++i) {
    auto device_address = AnfAlgo::GetMutableWorkspaceAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_) {
      mem_manager_->FreeMemFromMemPool(device_address);
    }
  }
}

DeviceAddressPtr GPUKernelRuntime::GetPrevNodeMutableOutputAddr(const AnfNodePtr &node, size_t i, bool skip_nop_node) {
  if (!enable_relation_cache_) {
    return AnfAlgo::GetPrevNodeMutableOutputAddr(node, i, skip_nop_node);
  }

  auto &addr_cache = skip_nop_node ? prev_node_mut_output_addr_cache_ : prev_node_mut_output_addr_skip_nop_node_cache_;
  std::unordered_map<AnfNodePtr, std::vector<session::KernelWithIndex>>::iterator addr_iter;
  if (auto iter = addr_cache.find(node); iter == addr_cache.end()) {
    addr_iter = addr_cache.insert({node, {common::AnfAlgo::GetInputTensorNum(node), {nullptr, 0}}}).first;
  } else {
    addr_iter = iter;
  }

  if (addr_iter->second[i].first == nullptr) {
    addr_iter->second[i] = common::AnfAlgo::GetPrevNodeOutput(node, i, skip_nop_node);
  }

  session::KernelWithIndex prev_node_with_index = addr_iter->second[i];
  auto kernel_info = dynamic_cast<device::KernelInfo *>(prev_node_with_index.first->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  auto addr = kernel_info->GetMutableOutputAddr(prev_node_with_index.second);

  return addr;
}

DeviceAddressPtr GPUKernelRuntime::GetMutableOutputAddr(const AnfNodePtr &node, size_t i, bool skip_nop_node) {
  if (!enable_relation_cache_) {
    return AnfAlgo::GetMutableOutputAddr(node, i, skip_nop_node);
  }

  auto &addr_cache = skip_nop_node ? mut_output_addr_cache_ : mut_output_addr_skip_nop_node_cache_;
  std::unordered_map<AnfNodePtr, std::vector<DeviceAddressPtr>>::iterator addr_iter;
  if (auto iter = addr_cache.find(node); iter == addr_cache.end()) {
    auto kernel_mod = AnfAlgo::GetKernelMod(node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    addr_iter = addr_cache.insert({node, {output_sizes.size(), nullptr}}).first;
  } else {
    addr_iter = iter;
  }

  auto &now_addr = addr_iter->second[i];
  if (now_addr == nullptr) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(node, i, skip_nop_node);
    now_addr = device_address;
  } else {
    if (addr_state_.count(now_addr) > 0) {
      addr_state_.erase(now_addr);
      auto device_address = AnfAlgo::GetMutableOutputAddr(node, i, skip_nop_node);
      now_addr = device_address;
    }
  }

  return now_addr;
}

session::KernelWithIndex GPUKernelRuntime::GetPrevNodeOutput(const AnfNodePtr &node, size_t i) {
  if (!enable_relation_cache_) {
    return common::AnfAlgo::GetPrevNodeOutput(node, i);
  }

  std::unordered_map<AnfNodePtr, std::vector<session::KernelWithIndex>>::iterator addr_iter;
  if (auto iter = prev_node_output_cache_.find(node); iter == prev_node_output_cache_.end()) {
    addr_iter = prev_node_output_cache_.insert({node, {common::AnfAlgo::GetInputTensorNum(node), {nullptr, 0}}}).first;
  } else {
    addr_iter = iter;
  }

  if (addr_iter->second[i].first == nullptr) {
    auto kernel_with_index = common::AnfAlgo::GetPrevNodeOutput(node, i);
    addr_iter->second[i] = kernel_with_index;
  }

  return addr_iter->second[i];
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
