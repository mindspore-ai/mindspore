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

#include "device/gpu/gpu_kernel_runtime.h"
#include "device/gpu/gpu_device_address.h"
#include "device/gpu/cuda_driver.h"
#include "device/gpu/gpu_buffer_mgr.h"
#include "device/gpu/gpu_device_manager.h"
#include "device/gpu/gpu_memory_allocator.h"
#include "device/gpu/distribution/collective_init.h"
#include "utils/convert_utils.h"
#include "utils/context/ms_context.h"
#include "device/kernel_runtime_manager.h"
#include "device/gpu/gpu_common.h"
#include "common/utils.h"
#include "device/gpu/gpu_memory_manager.h"
#include "kernel/common_utils.h"
#include "device/gpu/gpu_memory_copy_manager.h"

namespace mindspore {
namespace device {
namespace gpu {
using mindspore::device::memswap::MemSwapManager;
using mindspore::device::memswap::SwapKind;
bool GPUKernelRuntime::SyncStream() { return GPUDeviceManager::GetInstance().SyncStream(stream_); }

bool GPUKernelRuntime::Init() {
  if (device_init_ == true) {
    return true;
  }
  auto ret = InitDevice();
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

  // destroy remaining memory swap events and free host memory
  for (auto &item : mem_swap_map_) {
    auto &mem_swap_manager = item.second;
    MS_EXCEPTION_IF_NULL(mem_swap_manager);
    if (mem_swap_manager->trigger_swap()) {
      mem_swap_manager->ClearSwapQueue();
      mem_swap_manager->ReleaseHostPinnedMem();
    }
  }

  GPUDeviceManager::GetInstance().ReleaseDevice();
  if (mem_manager_ != nullptr) {
    mem_manager_->FreeDeviceMemory();
  }
  kernel::KernelMeta::GetInstance()->RemoveKernelCache();
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
    InitKernelOutputAddress(graph);
  } else {
    AssignDynamicMemory(graph);
  }
}

bool GPUKernelRuntime::Run(session::KernelGraph *graph) {
  bool ret = true;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_dynamic_mem = context_ptr->enable_dynamic_mem_pool();
  bool is_enable_pynative_infer = context_ptr->enable_pynative_infer();
  auto iter = mem_swap_map_.find(graph);
  if (iter == mem_swap_map_.end()) {
    GPUMemCopyManagerPtr gpu_mem_copy_manager = std::make_shared<GPUMemCopyManager>();
    iter = mem_swap_map_.emplace(graph, std::make_shared<MemSwapManager>(gpu_mem_copy_manager)).first;
  }
  mem_swap_manager_ = iter->second;
  MS_EXCEPTION_IF_NULL(mem_swap_manager_);
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  if (is_enable_dynamic_mem && !is_enable_pynative_infer) {
    while (!LaunchKernelDynamic(graph)) {
      ClearKernelOutputAddress(graph);
      if (!mem_swap_manager_->mem_swap_init()) {
        mem_swap_manager_->Init(graph);
      }
      if (!mem_swap_manager_->RetreatSwapInfo()) {
        return false;
      }
    }
  } else {
    ret = LaunchKernel(graph);
  }
  (void)gettimeofday(&end_time, nullptr);
  const uint64_t kUSecondInSecond = 1000000;
  uint64_t cost = kUSecondInSecond * static_cast<uint64_t>(end_time.tv_sec - start_time.tv_sec);
  cost += static_cast<uint64_t>(end_time.tv_usec - start_time.tv_usec);
  MS_LOG(DEBUG) << "kernel runtime run graph in " << cost << " us";
  return ret;
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
  auto graph_id = graph->graph_id();
  mem_reuse_util_map_[graph_id] = mem_reuse_util_ptr;
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

void GPUKernelRuntime::ClearKernelOutputAddress(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    auto output_sizes = kernel_mod->GetOutputSizeList();
    for (size_t i = 0; i < output_sizes.size(); ++i) {
      if (!AnfAlgo::OutputAddrExist(kernel, i)) {
        continue;
      }

      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
      if (device_address->ptr_) {
        mem_manager_->FreeMemFromMemPool(device_address);
      }
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
}

bool GPUKernelRuntime::LaunchKernelDynamic(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto graph_id = graph->graph_id();
  auto mem_reuse_util_ptr = mem_reuse_util_map_[graph_id];
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  // Reset the reference count.
  mem_reuse_util_ptr->ResetDynamicUsedRefCount();
  // The inputs and outputs memory of communication kernel need be continuous, so separate processing.
  AllocCommunicationOpDynamicRes(graph);

  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    auto ret = AllocKernelDynamicRes(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
    if (!ret) {
      return false;
    }
    if (!kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, stream_)) {
      MS_LOG(EXCEPTION) << "Launch kernel failed.";
    }
    FreeKernelDynamicRes(kernel, kernel_workspaces, graph_id);

    if (mem_swap_manager_->trigger_swap() && mem_swap_manager_->QueryKernelTriggerSwap(kernel)) {
      CHECK_OP_RET_WITH_EXCEPT(SyncStream(), "SyncStream failed.");
      if (!AddMemSwapTask(kernel)) {
        return false;
      }
    }

    if (mem_swap_manager_->trigger_swap()) {
      mem_swap_manager_->SyncMemCopyStream(SwapKind::kDeviceToHost);
    }
  }

  CHECK_OP_RET_WITH_EXCEPT(SyncStream(), "SyncStream failed.");
  if (mem_swap_manager_->trigger_swap()) {
    mem_swap_manager_->ClearSwapQueue();
  }
  return true;
}

bool GPUKernelRuntime::AddMemSwapTask(const AnfNodePtr &kernel) {
  auto &mem_swap_info_list = mem_swap_manager_->QueryKernelMemSwapInfo(kernel);
  for (auto &mem_swap_info : mem_swap_info_list) {
    auto &kernel_exec_info = mem_swap_manager_->SearchKernelExecutionInfo(mem_swap_info.kernel_);
    const HostAddress &host_address = kernel_exec_info.host_addrs_[mem_swap_info.output_idx_];
    auto device_address = AnfAlgo::GetMutableOutputAddr(mem_swap_info.kernel_, mem_swap_info.output_idx_);

    if (mem_swap_info.swap_kind_ == SwapKind::kDeviceToHost) {
      mem_swap_manager_->AddMemSwapTask(SwapKind::kDeviceToHost, device_address, host_address);
    } else if (mem_swap_info.swap_kind_ == SwapKind::kHostToDevice) {
      auto status = device_address->status();
      if (status == DeviceAddressStatus::kInDeviceToHost) {
        mem_swap_manager_->InsertSwapInBlackList(device_address->ptr_);
        device_address->set_status(DeviceAddressStatus::kInDevice);
      } else if (status == DeviceAddressStatus::kInHost) {
        if (!device_address->ptr_ && !AttemptMallocMem(device_address, device_address->size_)) {
          return false;
        }
        if (!mem_swap_manager_->FindInSwapInBlackList(device_address->ptr_)) {
          mem_swap_manager_->AddMemSwapTask(SwapKind::kHostToDevice, device_address, host_address);
        }
      }
    }
  }
  return true;
}

bool GPUKernelRuntime::AttemptMallocMem(const DeviceAddressPtr &device_address, size_t size) {
  auto ret = mem_manager_->MallocMemFromMemPool(device_address, size);
  if (!ret) {
    if (!mem_swap_manager_->trigger_swap()) {
      return false;
    }

    mem_swap_manager_->SyncMemCopyStream(SwapKind::kDeviceToHost);
    while (auto device_address_swap_out = mem_swap_manager_->UpdateSwapQueue(SwapKind::kDeviceToHost)) {
      if (!mem_swap_manager_->FindInSwapInBlackList(device_address_swap_out->ptr_) && device_address_swap_out->ptr_) {
        device_address_swap_out->set_status(DeviceAddressStatus::kInHost);
        mem_manager_->FreeMemFromMemPool(device_address_swap_out);
      }
    }

    ret = mem_manager_->MallocMemFromMemPool(device_address, size);
    if (!ret) {
      return false;
    }
  }
  return true;
}

void *GPUKernelRuntime::AttemptMallocMem(size_t size) {
  auto device_ptr = mem_manager_->MallocMemFromMemPool(size);
  if (!device_ptr) {
    if (!mem_swap_manager_->trigger_swap()) {
      return nullptr;
    }

    mem_swap_manager_->SyncMemCopyStream(SwapKind::kDeviceToHost);
    while (auto device_address_swap_out = mem_swap_manager_->UpdateSwapQueue(SwapKind::kDeviceToHost)) {
      if (!mem_swap_manager_->FindInSwapInBlackList(device_address_swap_out->ptr_) && device_address_swap_out->ptr_) {
        device_address_swap_out->set_status(DeviceAddressStatus::kInHost);
        mem_manager_->FreeMemFromMemPool(device_address_swap_out);
      }
    }

    device_ptr = mem_manager_->MallocMemFromMemPool(size);
    if (!device_ptr) {
      return nullptr;
    }
  }
  return device_ptr;
}

bool GPUKernelRuntime::AllocKernelDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                             const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs,
                                             AddressPtrList *kernel_workspaces, AddressPtrList *kernel_outputs) {
  if (!AllocKernelInputDynamicRes(kernel, kernel_inputs)) {
    return false;
  }
  if (!AllocKernelOutputDynamicRes(kernel_mod, kernel, kernel_outputs)) {
    return false;
  }
  if (!AllocKernelWorkspaceDynamicRes(kernel_mod, kernel, kernel_workspaces)) {
    return false;
  }
  return true;
}

bool GPUKernelRuntime::AllocKernelInputDynamicRes(const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (mem_swap_manager_->trigger_swap()) {
      while (auto device_address_swap_in = mem_swap_manager_->UpdateSwapQueue(SwapKind::kHostToDevice)) {
        device_address_swap_in->set_status(DeviceAddressStatus::kInDevice);
      }

      auto status = device_address->status();
      switch (status) {
        case DeviceAddressStatus::kInDevice:
          break;
        case DeviceAddressStatus::kInHost:
          break;
        case DeviceAddressStatus::kInDeviceToHost: {
          mem_swap_manager_->InsertSwapInBlackList(device_address->ptr_);
          device_address->set_status(DeviceAddressStatus::kInDevice);
          break;
        }
        case DeviceAddressStatus::kInHostToDevice: {
          while (device_address->status() != DeviceAddressStatus::kInDevice) {
            while (auto device_address_swap_in = mem_swap_manager_->UpdateSwapQueue(SwapKind::kHostToDevice)) {
              device_address_swap_in->set_status(DeviceAddressStatus::kInDevice);
            }
          }
          break;
        }
        default:
          MS_LOG(ERROR) << "Invaild device address status";
          return false;
      }
    }
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
                                                   const mindspore::AnfNodePtr &kernel,
                                                   AddressPtrList *kernel_outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  if (mem_swap_manager_->trigger_swap()) {
    while (auto device_address_swap_out = mem_swap_manager_->UpdateSwapQueue(SwapKind::kDeviceToHost)) {
      if (!mem_swap_manager_->FindInSwapInBlackList(device_address_swap_out->ptr_) && device_address_swap_out->ptr_) {
        device_address_swap_out->set_status(DeviceAddressStatus::kInHost);
        mem_manager_->FreeMemFromMemPool(device_address_swap_out);
      }
    }
  }
  auto output_sizes = kernel_mod.GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    if (device_address->ptr_ == nullptr && !AttemptMallocMem(device_address, output_sizes[i])) {
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
                                                      AddressPtrList *kernel_workspaces) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto workspace_sizes = kernel_mod.GetWorkspaceSizeList();
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    if (workspace_sizes[i] == 0) {
      kernel_workspaces->emplace_back(nullptr);
      continue;
    }
    auto device_ptr = AttemptMallocMem(workspace_sizes[i]);
    if (!device_ptr) {
      return false;
    }
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_ptr;
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
  MS_EXCEPTION_IF_NULL(mem_manager_);
  bool is_need_alloc_memory = false;
  bool is_need_free_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  DeviceAddressPtrList addr_list;
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
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
  MS_EXCEPTION_IF_NULL(mem_manager_);
  bool is_need_alloc_memory = false;
  bool is_need_free_memory = false;
  size_t total_size = 0;
  std::vector<size_t> size_list;
  DeviceAddressPtrList addr_list;
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
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

void GPUKernelRuntime::FreeKernelDynamicRes(const mindspore::AnfNodePtr &kernel,
                                            const AddressPtrList &kernel_workspaces, uint32_t graph_id) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  auto mem_reuse_util_ptr = mem_reuse_util_map_[graph_id];
  MS_EXCEPTION_IF_NULL(mem_reuse_util_ptr);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (AnfAlgo::GetCNodeName(kernel) == kAllReduceOpName) {
    return;
  }
  // Free the input of kernel by reference count.
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto kernel_ref_count_ptr = mem_reuse_util_ptr->GetKernelInputRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    kernel_ref_count_ptr->ref_count_dynamic_use_--;
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ < 0) {
      MS_LOG(EXCEPTION) << "Check dynamic reference count failed.";
    }
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
      mem_manager_->FreeMemFromMemPool(device_address);
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
  // Free the output of kernel, if output has no reference.
  for (size_t i = 0; i < AnfAlgo::GetOutputTensorNum(kernel); ++i) {
    auto kernel_ref_count_ptr = mem_reuse_util_ptr->GetRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
      mem_manager_->FreeMemFromMemPool(device_address);
      device_address->set_status(DeviceAddressStatus::kInDevice);
    }
  }
  // Free the workspace of kernel.
  for (size_t i = 0; i < kernel_workspaces.size(); ++i) {
    auto workspace = kernel_workspaces[i];
    if (workspace != nullptr) {
      MS_EXCEPTION_IF_NULL(workspace->addr);
      mem_manager_->FreeMemFromMemPool(workspace->addr);
      workspace->addr = nullptr;
    }
  }
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
