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
#include "runtime/device/gpu/gpu_kernel_runtime.h"
#include <algorithm>
#include "runtime/device/gpu/gpu_device_address.h"
#include "runtime/device/gpu/cuda_driver.h"
#include "runtime/device/gpu/gpu_buffer_mgr.h"
#include "runtime/device/gpu/gpu_device_manager.h"
#include "runtime/device/gpu/gpu_memory_allocator.h"
#include "runtime/device/gpu/distribution/collective_init.h"
#include "utils/convert_utils.h"
#include "utils/ms_context.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/device/gpu/gpu_common.h"
#include "utils/ms_utils.h"
#include "runtime/device/gpu/gpu_memory_manager.h"
#include "backend/kernel_compiler/common_utils.h"
#include "runtime/device/gpu/gpu_memory_copy_manager.h"
#include "common/trans.h"
#include "ir/dtype.h"
#include "profiler/device/gpu/gpu_profiling.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
using mindspore::device::memswap::MemSwapInfoSet;
using mindspore::device::memswap::MemSwapManager;
using mindspore::device::memswap::SwapKind;
static const size_t PARAMETER_OUTPUT_INDEX = 0;
bool GPUKernelRuntime::SyncStream() { return GPUDeviceManager::GetInstance().SyncStream(stream_); }

bool GPUKernelRuntime::Init() {
  if (device_init_ == true) {
    GPUMemoryAllocator::GetInstance().CheckMaxDeviceMemory();
    return true;
  }
  bool ret = false;
#ifdef ENABLE_DUMP_E2E
  ret = SetDumpConf();
  if (!ret) {
    MS_LOG(INFO) << "No dump conf to set!";
  }
#endif

  ret = InitDevice();
  if (!ret) {
    MS_LOG(ERROR) << "InitDevice error.";
    return ret;
  }
  mem_manager_ = std::make_shared<GPUMemoryManager>();
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->MallocDeviceMemory();
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  bool collective_inited = CollectiveInitializer::instance().collective_inited();
  if (collective_inited && collective_handle_ != nullptr) {
    auto init_nccl_comm_funcptr =
      reinterpret_cast<InitNCCLComm>(dlsym(const_cast<void *>(collective_handle_), "InitNCCLComm"));
    MS_EXCEPTION_IF_NULL(init_nccl_comm_funcptr);
    (*init_nccl_comm_funcptr)();
  }
  device_init_ = true;
  return ret;
}

#ifdef ENABLE_DUMP_E2E
namespace {
void DumpOutput(mindspore::session::KernelGraph *graph, const string &dump_path, DumpConfPtr dump_conf,
                Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(dump_conf);
  bool trans_flag = dump_conf->trans_flag();
  const auto &apply_kernels = graph->execution_order();
  for (const auto &node : apply_kernels) {
    MS_EXCEPTION_IF_NULL(node);
    auto node_name = AnfAlgo::GetCNodeName(node);
    std::string kernel_name = node->fullname_with_scope();
    if (!dump_conf->IsKernelNeedDump(kernel_name)) {
      continue;
    }
    const std::string strsrc = "/";
    const std::string strdst = "--";
    std::string::size_type pos = 0;
    std::string::size_type srclen = strsrc.size();
    std::string::size_type dstlen = strdst.size();
    while ((pos = kernel_name.find(strsrc, pos)) != std::string::npos) {
      kernel_name.replace(pos, srclen, strdst);
      pos += dstlen;
    }
    auto output_size = AnfAlgo::GetOutputTensorNum(node);
    for (size_t j = 0; j < output_size; ++j) {
      auto addr = AnfAlgo::GetOutputAddr(node, j);
      TypeId addr_type_id = addr->type_id();
      std::string addr_format = addr->format();
      std::vector<int> int_shapes;
      if (trans_flag) {
        int_shapes = trans::GetRuntimePaddingShape(node, j);
      } else {
        auto shape = AnfAlgo::GetOutputDeviceShape(node, j);
        (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                             [](size_t inner_item) { return SizeToInt(inner_item); });
      }

      auto type = AnfAlgo::GetOutputInferDataType(node, j);

      auto format = kOpFormat_DEFAULT;
      string filepath = dump_path + '/' + kernel_name + '_' + "output_" + std::to_string(j);

      DebugServices *debug_services = debugger->debug_services();
      TensorLoader *tensor_loader = debug_services->tensor_loader();
      std::string original_kernel_name = node->fullname_with_scope();
      size_t slot = j;
      auto ret = tensor_loader->DumpTensorToFile(original_kernel_name, trans_flag, filepath, format, int_shapes, type,
                                                 addr_type_id, addr_format, slot);

      if (!ret) {
        std::string error = "DumpTensorToFile Failed: flag:" + std::to_string(trans_flag) + ", path:" + filepath +
                            ", host_format:" + format + ".!";
      }
    }
  }
}

void DumpParameters(mindspore::session::KernelGraph *graph, const string &dump_path, DumpConfPtr dump_conf,
                    Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(dump_conf);
  bool trans_flag = dump_conf->trans_flag();
  const auto &parameters = graph->inputs();
  for (auto &item : parameters) {
    if (!item->isa<Parameter>()) {
      continue;
    }
    std::string parameter_name = item->fullname_with_scope();
    if (!dump_conf->IsKernelNeedDump(parameter_name)) {
      continue;
    }
    auto addr = AnfAlgo::GetOutputAddr(item, PARAMETER_OUTPUT_INDEX);
    TypeId addr_type_id = addr->type_id();
    std::string addr_format = addr->format();
    std::vector<int> int_shapes;
    if (trans_flag) {
      int_shapes = trans::GetRuntimePaddingShape(item, PARAMETER_OUTPUT_INDEX);
    } else {
      auto shape = AnfAlgo::GetOutputDeviceShape(item, PARAMETER_OUTPUT_INDEX);
      (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                           [](size_t inner_item) { return SizeToInt(inner_item); });
    }

    auto type = AnfAlgo::GetOutputInferDataType(item, PARAMETER_OUTPUT_INDEX);

    auto format = kOpFormat_DEFAULT;
    string filepath = dump_path + '/' + parameter_name + '_' + "output_0";

    DebugServices *debug_services = debugger->debug_services();
    TensorLoader *tensor_loader = debug_services->tensor_loader();
    std::string original_kernel_name = parameter_name;
    size_t slot = 0;
    auto ret = tensor_loader->DumpTensorToFile(original_kernel_name, trans_flag, filepath, format, int_shapes, type,
                                               addr_type_id, addr_format, slot);

    if (!ret) {
      std::string error = "DumpTensorToFile Failed: flag:" + std::to_string(trans_flag) + ", path:" + filepath +
                          ", host_format:" + format + ".!";
    }
  }
}
}  // namespace

bool GPUKernelRuntime::DumpData(mindspore::session::KernelGraph *graph, Debugger *debugger) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_LOG(INFO) << "Start dump step";
  DumpConfPtr dump_conf = GetDumpConf();
  MS_EXCEPTION_IF_NULL(dump_conf);
  dump_conf->UpdataCurIter();
  bool dump_flag = dump_conf->dump_enable();
  if (!dump_flag) {
    MS_LOG(INFO) << "Dump flag is disable, pass dump step";
    return true;
  }
  uint32_t cur_iter = dump_conf->cur_iter();
  if (dump_conf->dump_iter() != 0) {
    if (cur_iter != dump_conf->dump_iter()) {
      return true;
    }
  }
  MS_LOG(INFO) << "Cur iter is " << cur_iter;
  std::string net_name = dump_conf->dump_net_name();
  std::string iterator = std::to_string(cur_iter);
  std::string dump_path = dump_conf->dump_path();
  if (dump_path.back() == '/') {
    dump_path = dump_path + net_name + '/' + iterator;
  } else {
    dump_path = dump_path + '/' + net_name + '/' + iterator;
  }

  // dump output
  DumpOutput(graph, dump_path, dump_conf, debugger);
  // dump parameters
  DumpParameters(graph, dump_path, dump_conf, debugger);

  return true;
}
#endif

#ifdef ENABLE_DEBUGGER
namespace {
void LoadKernelData(Debugger *debugger, const CNodePtr &kernel,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_inputs,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_workspaces,
                    const std::vector<mindspore::kernel::AddressPtr> &kernel_outputs, int exec_order, void *stream_ptr,
                    bool dump_enabled) {
  // check if we should read the kernel data
  bool read_data = false;
  std::string kernel_name = kernel->fullname_with_scope();
  if (debugger) {
    debugger->SetCurNode(kernel_name);
    if (dump_enabled) {
      read_data = true;
    } else if (debugger->debugger_enabled()) {
      read_data = debugger->ReadNodeDataRequired();
    }
  }

  if (!read_data) {
    return;
  }

  // get inputs
  auto input_size = AnfAlgo::GetInputTensorNum(kernel);
  for (size_t j = 0; j < input_size; ++j) {
    auto input_kernel = kernel->input(j + 1);
    std::string input_kernel_name = input_kernel->fullname_with_scope();
    auto addr = kernel_inputs[j];
    auto type = AnfAlgo::GetOutputInferDataType(input_kernel, PARAMETER_OUTPUT_INDEX);
    auto format = kOpFormat_DEFAULT;
    auto gpu_addr = std::make_unique<GPUDeviceAddress>(addr->addr, addr->size, format, type);
    string input_tensor_name = input_kernel_name + ':' + "0";
    std::vector<int> int_shapes;
    auto shape = AnfAlgo::GetOutputDeviceShape(input_kernel, PARAMETER_OUTPUT_INDEX);
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                         [](size_t inner_item) { return SizeToInt(inner_item); });
    auto ret = gpu_addr->LoadMemToHost(input_tensor_name, exec_order, format, int_shapes, type, 0, debugger, false);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << input_tensor_name << ", host_format:" << format << ".!";
    }
  }

  // get outputs
  auto output_size = AnfAlgo::GetOutputTensorNum(kernel);
  for (size_t j = 0; j < output_size; ++j) {
    auto addr = kernel_outputs[j];
    auto type = AnfAlgo::GetOutputInferDataType(kernel, j);
    auto format = kOpFormat_DEFAULT;
    auto gpu_addr = std::make_unique<GPUDeviceAddress>(addr->addr, addr->size, format, type);
    string tensor_name = kernel_name + ':' + std::to_string(j);
    std::vector<int> int_shapes;
    auto shape = AnfAlgo::GetOutputDeviceShape(kernel, j);
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                         [](size_t inner_item) { return SizeToInt(inner_item); });
    auto ret = gpu_addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, j, debugger, false);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
    }
  }

  debugger->PostExecuteNode();
}

void UpdateStepNum(Debugger *debugger, bool dump_enabled) {
  if (debugger && (debugger->debugger_enabled() || dump_enabled)) {
    auto cur_step_num = debugger->step_num();
    cur_step_num = cur_step_num + 1;
    debugger->SetStepNum(cur_step_num);
  }
}

void LoadParameters(const session::KernelGraph *graph, Debugger *debugger, bool dump_enabled) {
  MS_EXCEPTION_IF_NULL(graph);
  if (!(debugger && dump_enabled)) {
    return;
  }
  const auto &parameters = graph->inputs();
  // for parameters, set its execution order to be 0;
  int exec_order = 0;
  for (auto &item : parameters) {
    if (!item->isa<Parameter>()) {
      continue;
    }
    std::string parameter_name = item->fullname_with_scope();
    auto addr = AnfAlgo::GetOutputAddr(item, PARAMETER_OUTPUT_INDEX);
    auto type = AnfAlgo::GetOutputInferDataType(item, PARAMETER_OUTPUT_INDEX);
    auto format = kOpFormat_DEFAULT;
    string tensor_name = parameter_name + ':' + "0";
    auto gpu_addr = dynamic_cast<const mindspore::device::gpu::GPUDeviceAddress *>(addr);
    std::vector<int> int_shapes;
    auto shape = AnfAlgo::GetOutputDeviceShape(item, PARAMETER_OUTPUT_INDEX);
    (void)std::transform(shape.begin(), shape.end(), std::back_inserter(int_shapes),
                         [](size_t inner_item) { return SizeToInt(inner_item); });
    auto ret = gpu_addr->LoadMemToHost(tensor_name, exec_order, format, int_shapes, type, 0, debugger, true);
    if (!ret) {
      MS_LOG(ERROR) << "LoadMemToHost:"
                    << ", tensor_name:" << tensor_name << ", host_format:" << format << ".!";
    }
  }
}

void ClearCurrentData(Debugger *debugger, bool dump_enabled) {
  if (debugger && (debugger->debugger_enabled() || dump_enabled)) {
    DebugServices *debug_services = debugger->debug_services();
    TensorLoader *tensor_loader = debug_services->tensor_loader();
    tensor_loader->EmptyCurrentTensor();
  }
}
}  // namespace
#endif

DeviceAddressPtr GPUKernelRuntime::CreateDeviceAddress(void *device_ptr, size_t device_size, const string &format,
                                                       TypeId type_id) {
  return std::make_shared<GPUDeviceAddress>(device_ptr, device_size, format, type_id);
}

bool GPUKernelRuntime::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }
  const void *collective_handle_ = CollectiveInitializer::instance().collective_handle();
  bool collective_inited = CollectiveInitializer::instance().collective_inited();
  if (collective_inited && collective_handle_ != nullptr) {
    auto get_local_rank_funcptr =
      reinterpret_cast<GetLocalRankId>(dlsym(const_cast<void *>(collective_handle_), "local_rank_id"));
    MS_EXCEPTION_IF_NULL(get_local_rank_funcptr);
    device_id_ = IntToUint((*get_local_rank_funcptr)());
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
  return true;
}

void GPUKernelRuntime::ReleaseDeviceRes() {
  // For dataset mode.
  if (GpuBufferMgr::GetInstance().IsInit()) {
    if (!GpuBufferMgr::GetInstance().IsClosed()) {
      if (!GpuBufferMgr::GetInstance().CloseNotify()) {
        MS_LOG(EXCEPTION) << "Could not close gpu data queue.";
      }
    }
    CHECK_OP_RET_WITH_EXCEPT(GpuBufferMgr::GetInstance().Destroy(), "Could not destroy gpu data queue.");
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
    mem_manager_->FreeDeviceMemory();
  }

  kernel::KernelMeta *bin_map = kernel::KernelMeta::GetInstance();
  MS_EXCEPTION_IF_NULL(bin_map);
  bin_map->RemoveKernelCache();
}

void GPUKernelRuntime::AssignMemory(session::KernelGraph *graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();
  AssignStaticMemoryInput(graph);
  AssignStaticMemoryValueNode(graph);
  bool is_enable_dynamic_mem = context_ptr->enable_dynamic_mem_pool();
  if (is_enable_dynamic_mem) {
    // Use the dynamic memory pool.
    InitKernelRefCount(graph);
    InitMemorySwapInfo(graph);
    InitKernelOutputAddress(graph);
    InitKernelWorkspaceAddress(graph);
    SaveGraphOutputNode(graph);
  } else {
    AssignDynamicMemory(graph);
  }
}

bool GPUKernelRuntime::Run(session::KernelGraph *graph, Debugger *debugger) {
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  bool ret = true;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_dynamic_mem = context_ptr->enable_dynamic_mem_pool();
  bool is_enable_pynative_infer = context_ptr->enable_pynative_infer();
  if (is_enable_dynamic_mem && !is_enable_pynative_infer) {
    auto graph_id = graph->graph_id();
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

    ret = RunOneStep(graph, debugger);
  } else {
    ret = LaunchKernel(graph);
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(DEBUG) << "GPU kernel runtime run graph in " << cost << " us";
  return ret;
}

bool GPUKernelRuntime::RunOneStep(const session::KernelGraph *graph, Debugger *debugger) {
  bool ret = true;
  auto graph_id = graph->graph_id();
  if (!is_first_step_map_[graph_id]) {
    // Normally run graph
    ret = LaunchKernelDynamic(graph, debugger);
  } else {
    // Mock run first step
    ret = LaunchKernelDynamic(graph, debugger, true, false);
    if (ret) {
      // Normally run graph
      ret = LaunchKernelDynamic(graph, debugger);
    } else {
      // Trigger memory swap
      ret = SearchMemSwapScheme(graph, debugger);
    }
    is_first_step_map_[graph_id] = false;
  }
  return ret;
}

bool GPUKernelRuntime::SearchMemSwapScheme(const session::KernelGraph *graph, Debugger *debugger) {
  MS_LOG(WARNING) << "Run out of memory and try memory swapping, it may take some time, please wait a moment.";
  bool ret = false;
  ClearKernelOldOutputAndWorkspace(graph);
  if (!mem_swap_manager_->mem_swap_init()) {
    if (!mem_swap_manager_->Init(graph)) {
      return false;
    }
  }

  while (!ret) {
    if (!mem_swap_manager_->RetreatSwapInfo()) {
      return false;
    }
    ret = LaunchKernelDynamic(graph, debugger, true, false);
    if (!ret) {
      ClearKernelOldOutputAndWorkspace(graph);
    }
  }
  mem_swap_manager_->AssignHostMemory();

  // Time profiling
  ret = LaunchKernelDynamic(graph, debugger, false, true);
  if (!ret) {
    return ret;
  }
  return RefineMemSwapScheme(graph, debugger);
}

bool GPUKernelRuntime::RefineMemSwapScheme(const session::KernelGraph *graph, Debugger *debugger) {
  MS_LOG(WARNING) << "Refine memory swap scheme, it may take some time, please wait a moment.";
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
        ret = LaunchKernelDynamic(graph, debugger, true, false);
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
  // Can't free the device address of summary nodes, so set the reference count of summary nodes specially.
  mem_reuse_util_ptr->SetSummaryNodesRefCount();
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
  const auto &output_nodes = AnfAlgo::GetAllOutput(graph->output(), {prim::kPrimTupleGetItem});
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

bool GPUKernelRuntime::LaunchKernelDynamic(const session::KernelGraph *graph, Debugger *debugger, bool mock,
                                           bool profiling) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(mem_reuse_util_);
  // Reset the reference count.
  mem_reuse_util_->ResetDynamicUsedRefCount();
  // The inputs and outputs memory of communication kernel need be continuous, so separate processing.
  AllocCommunicationOpDynamicRes(graph);

#ifdef ENABLE_DEBUGGER
  bool dump_enabled = GPUKernelRuntime::DumpDataEnabledIteration();
  if (!mock) {
    UpdateStepNum(debugger, dump_enabled);
  }
#endif
  auto &kernels = graph->execution_order();
  int exec_order = 1;

  auto profiler_inst = profiler::gpu::GPUProfiler::GetInstance();
  if (profiler_inst == nullptr) {
    MS_LOG(ERROR) << "gpu profiler instance is nullptr";
  }

  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    auto ret = AllocKernelDynamicRes(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs, mock);
    if (!ret) {
#ifdef ENABLE_DEBUGGER
      if (!mock) {
        // invalidate current data collected by the debugger
        ClearCurrentData(debugger, dump_enabled);
      }
#endif
      return false;
    }
    if (!mock) {
      if (!profiling) {
        if (profiler_inst->GetEnableFlag()) {
          profiler_inst->OpDataProducerBegin(kernel->fullname_with_scope(), stream_);
        }
        CHECK_OP_RET_WITH_EXCEPT(kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_),
                                 "Launch kernel failed.");
        if (profiler_inst->GetEnableFlag()) {
          profiler_inst->OpDataProducerEnd();
          if (profiler_inst->GetSyncEnableFlag()) {
            CHECK_OP_RET_WITH_ERROR(SyncStream(), "Profiler SyncStream failed.");
          }
        }
      } else {
        LaunchKernelWithTimeProfiling(kernel, kernel_inputs, kernel_workspaces, kernel_outputs);
      }
#ifdef ENABLE_DEBUGGER
      // called once per kernel to collect the outputs to the kernel (does a SyncDeviceToHost)
      LoadKernelData(debugger, kernel, kernel_inputs, kernel_workspaces, kernel_outputs, exec_order, stream_,
                     dump_enabled);
#endif
    }
    exec_order = exec_order + 1;
    FreeKernelDynamicRes(kernel);
    if (!UpdateMemorySwapTask(kernel, mock, profiling)) {
#ifdef ENABLE_DEBUGGER
      if (!mock) {
        // invalidate current data collected by the debugger
        ClearCurrentData(debugger, dump_enabled);
      }
#endif
      return false;
    }
  }
  if (!mock) {
#ifdef ENABLE_DEBUGGER
    // collect weights and bias for dump mode
    LoadParameters(graph, debugger, dump_enabled);
#endif
    CHECK_OP_RET_WITH_EXCEPT(SyncStream(), "SyncStream failed.");
  }
  ClearSwapInfo(mock);
  return true;
}

void GPUKernelRuntime::LaunchKernelWithTimeProfiling(const AnfNodePtr &kernel, const AddressPtrList &inputs,
                                                     const AddressPtrList &workspace, const AddressPtrList &outputs) {
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  float cost_time = 0;
  DeviceEvent start = nullptr;
  DeviceEvent end = nullptr;
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateEvent(&start), "Failed to create event.");
  CHECK_OP_RET_WITH_EXCEPT(CudaDriver::CreateEvent(&end), "Failed to create event.");

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
  const MemSwapInfoSet &mem_swap_info_set = mem_swap_manager_->QueryKernelMemSwapInfo(kernel);
  for (auto &mem_swap_info : mem_swap_info_set) {
    auto need_swap_kernel = mem_swap_manager_->QueryKernelByTopoOrder(mem_swap_info.topo_order_);
    MS_EXCEPTION_IF_NULL(need_swap_kernel);
    const HostAddress &host_address =
      mem_swap_manager_->QueryKernelHostAddr(need_swap_kernel, mem_swap_info.output_idx_);
    auto device_address = AnfAlgo::GetMutableOutputAddr(need_swap_kernel, mem_swap_info.output_idx_, false);

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
      MS_LOG(EXCEPTION) << "Invaild device address status: " << status;
  }
}

void GPUKernelRuntime::UpdateHostSwapOutQueue(bool mock) {
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
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
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    DeviceAddressPtr device_address;
    if (mem_reuse_util_->is_all_nop_node()) {
      // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
      device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, false);
    } else {
      // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
      device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, true);
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
  auto output_sizes = kernel_mod.GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
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
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    if (AnfAlgo::IsCommunicationOp(kernel)) {
      AllocCommunicationOpInputDynamicRes(kernel);
      AllocCommunicationOpOutputDynamicRes(kernel);
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
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    DeviceAddressPtr device_address;
    if (mem_reuse_util_->is_all_nop_node()) {
      // Graph may be all nop nodes and not remove nop node, so this can not skip nop node.
      device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, false);
    } else {
      // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
      device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, true);
    }
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr) {
      is_need_alloc_memory = true;
    } else {
      is_need_free_memory = true;
    }
    total_size += device_address->size_;
    size_list.emplace_back(device_address->size_);
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
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
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

void GPUKernelRuntime::AllocCommunicationOpMemory(bool is_need_alloc_memory, bool is_need_free_memory,
                                                  const DeviceAddressPtrList addr_list, size_t total_size,
                                                  std::vector<size_t> size_list) {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (!is_need_alloc_memory) {
    return;
  }
  if (is_need_free_memory) {
    for (const auto &iter : addr_list) {
      MS_EXCEPTION_IF_NULL(iter);
      // Free the inputs/outputs of communication kernel which are not released.
      if (iter->ptr_ != nullptr) {
        mem_manager_->FreeMemFromMemPool(iter);
      }
    }
  }
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
  if (AnfAlgo::IsCommunicationOp(kernel)) {
    return;
  }
  // Free the input of kernel by reference count.
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
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
        device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, false);
      } else {
        // Graph may be "nop node + depend + node",  the input of node is the depend, so this case need skip nop node.
        device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i, true);
      }
      mem_manager_->FreeMemFromMemPool(device_address);
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
  // Free the output of kernel, if output has no reference.
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel); ++i) {
    auto kernel_ref_count_ptr = mem_reuse_util_->GetRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i, false);
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
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
