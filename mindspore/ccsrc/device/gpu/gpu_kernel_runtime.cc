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

namespace mindspore {
namespace device {
namespace gpu {
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

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  // If use the dynamic memory pool, then alloc the first memory block to init.
  if (context_ptr->enable_dynamic_mem_pool()) {
    auto device_addr = AllocTensorMemDynamic(1);
    if (!device_addr) {
      MS_LOG(ERROR) << "Dynamic memory pool init error.";
      return false;
    }
  } else {
    MallocDeviceMemory();
  }

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

void GPUKernelRuntime::MallocDeviceMemory() {
  // Need to reserve 20% space for dynamic memory
  const float init_gpu_mem_ratio = 0.8;
  size_t mem_size = FloatToSize(GPUMemoryAllocator::GetInstance().free_mem_size() * init_gpu_mem_ratio);
  auto alloc_size =
    GPUMemoryAllocator::GetInstance().AllocDeviceMem(mem_size, reinterpret_cast<void **>(&device_mem_base_));
  device_mem_size_ = alloc_size;
  static_mem_offset_ = device_mem_size_;
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
  GPUDeviceManager::GetInstance().ReleaseDevice();
  if (device_mem_base_ != nullptr) {
    if (!GPUMemoryAllocator::GetInstance().FreeDeviceMem(device_mem_base_)) {
      MS_LOG(EXCEPTION) << "Could not free gpu device memory.";
    }
  }
  GPUMemoryAllocator::GetInstance().ReleaseDeviceRes();
}

void GPUKernelRuntime::FreeHostMemory() { dynamic_mem_offset_ = 0; }

void *GPUKernelRuntime::AllocTensorMemDynamic(size_t size) {
  return GPUMemoryAllocator::GetInstance().AllocTensorMem(size);
}

void GPUKernelRuntime::FreeTensorMemDynamic(void *device_ptr) {
  GPUMemoryAllocator::GetInstance().FreeTensorMem(device_ptr);
}

void GPUKernelRuntime::AssignMemory(session::KernelGraph *graph) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  AssignStaticMemory(graph);
  bool is_enable_mem_reuse = context_ptr->enable_mem_reuse();
  bool is_enable_dynamic_mem = context_ptr->enable_dynamic_mem_pool();
  if (is_enable_dynamic_mem) {
    // Use the dynamic memory pool.
    InitKernelRefCount(graph);
    InitKernelOutputAddress(graph);
  } else if (is_enable_mem_reuse) {
    // Use the memory reuse.
    ReuseAssignDynamicMemory(graph);
  } else {
    // Normal way.
    AssignDynamicMemory(graph);
  }
}

bool GPUKernelRuntime::Run(session::KernelGraph *graph) {
  bool ret;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool is_enable_dynamic_mem = context_ptr->enable_dynamic_mem_pool();
  struct timeval start_time, end_time;
  (void)gettimeofday(&start_time, nullptr);
  if (is_enable_dynamic_mem) {
    ret = LaunchKernelDynamic(graph);
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

uint8_t *GPUKernelRuntime::MallocStaticMem(size_t size, bool) {
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->enable_dynamic_mem_pool()) {
    auto device_ptr = AllocTensorMemDynamic(size);
    MS_EXCEPTION_IF_NULL(device_ptr);
    return AddressOffset(device_ptr, 0);
  }

  auto align_size = GetCommonAlignSize(size);
  if (static_mem_offset_ < align_size) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  auto offset = static_mem_offset_ - align_size;
  if (dynamic_mem_offset_ > offset) {
    MS_LOG(EXCEPTION) << "Out of memory!!! total[" << device_mem_size_ << "](dynamic[" << total_dynamic_size_
                      << "] static[" << total_static_size_ << "])"
                      << " malloc [" << align_size << "] failed!";
  }
  total_static_size_ += align_size;
  static_mem_offset_ = offset;
  return device_mem_base_ + offset;
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
  // Can't free the device address of graph output, so set the reference count of graph output specially,
  mem_reuse_util_ptr->SetGraphOutputRefCount();
  mem_reuse_util_ptr_ = mem_reuse_util_ptr;
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

bool GPUKernelRuntime::LaunchKernelDynamic(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  // The inputs and outputs memory of communication kernel are special, so separate processing.
  AllocCommunicationOpDynamicRes(graph);

  auto &kernels = graph->execution_order();
  for (const auto &kernel : kernels) {
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    AddressPtrList kernel_inputs;
    AddressPtrList kernel_workspaces;
    AddressPtrList kernel_outputs;
    AllocKernelDynamicRes(*kernel_mod, kernel, &kernel_inputs, &kernel_workspaces, &kernel_outputs);
    if (!kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, reinterpret_cast<uintptr_t>(stream_))) {
      MS_LOG(ERROR) << "Launch kernel failed.";
      return false;
    }
    FreeKernelDynamicRes(kernel, kernel_workspaces);
  }

  if (!SyncStream()) {
    MS_LOG(ERROR) << "SyncStream failed.";
    return false;
  }
  return true;
}

void GPUKernelRuntime::AllocKernelDynamicRes(const mindspore::kernel::KernelMod &kernel_mod,
                                             const mindspore::AnfNodePtr &kernel, AddressPtrList *kernel_inputs,
                                             AddressPtrList *kernel_workspaces, AddressPtrList *kernel_outputs) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(kernel_inputs);
  MS_EXCEPTION_IF_NULL(kernel_workspaces);
  MS_EXCEPTION_IF_NULL(kernel_outputs);
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto device_address = AnfAlgo::GetPrevNodeOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(device_address->ptr_);
    kernel::AddressPtr input = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(input);
    input->addr = device_address->ptr_;
    input->size = device_address->size_;
    kernel_inputs->push_back(input);
  }

  auto output_sizes = kernel_mod.GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    auto device_ptr = device_address->ptr_;
    if (device_ptr == nullptr) {
      device_ptr = AllocTensorMemDynamic(output_sizes[i]);
      MS_EXCEPTION_IF_NULL(device_ptr);
      device_address->ptr_ = device_ptr;
    }
    kernel::AddressPtr output = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(output);
    output->addr = device_ptr;
    output->size = output_sizes[i];
    kernel_outputs->push_back(output);
  }

  auto workspace_sizes = kernel_mod.GetWorkspaceSizeList();
  for (size_t i = 0; i < workspace_sizes.size(); ++i) {
    if (workspace_sizes[i] == 0) {
      kernel_workspaces->emplace_back(nullptr);
      continue;
    }
    auto device_ptr = AllocTensorMemDynamic(workspace_sizes[i]);
    MS_EXCEPTION_IF_NULL(device_ptr);
    kernel::AddressPtr workspace = std::make_shared<kernel::Address>();
    MS_EXCEPTION_IF_NULL(workspace);
    workspace->addr = device_ptr;
    workspace->size = workspace_sizes[i];
    kernel_workspaces->push_back(workspace);
  }
}

void GPUKernelRuntime::AllocCommunicationOpDynamicRes(const session::KernelGraph *graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto &kernels = graph->execution_order();
  for (auto &kernel : kernels) {
    MS_EXCEPTION_IF_NULL(kernel);
    auto kernel_name = AnfAlgo::GetCNodeName(kernel);
    if (kernel_name == kAllReduceOpName) {
      AllocCommunicationOpInputDynamicRes(kernel);
      AllocCommunicationOpOutputDynamicRes(kernel);
      return;
    }
  }
}

void GPUKernelRuntime::AllocCommunicationOpInputDynamicRes(const mindspore::AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  // The reference count of communication kernel input is not 0.
  if (communication_op_input_ref_count_ != 0) {
    MS_LOG(ERROR) << "The reference count of communication kernel input is not 0.";
    return;
  }

  size_t total = 0;
  std::vector<std::pair<mindspore::device::DeviceAddress *, size_t>> addr_size;
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    // The inputs of communication kernel are not released.
    if ((i == 0) && (device_address->ptr_ != nullptr)) {
      MS_LOG(ERROR) << "The inputs of communication kernel are not released.";
      return;
    }
    auto output_size = device_address->size_;
    total += output_size;
    addr_size.emplace_back(device_address.get(), output_size);
  }

  auto device_mem_ptr = AllocTensorMemDynamic(total);
  MS_EXCEPTION_IF_NULL(device_mem_ptr);
  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(device_mem_ptr);
    communication_op_input_ref_count_++;
    device_mem_ptr = AddressOffset(device_mem_ptr, iter.second);
  }
}

void GPUKernelRuntime::AllocCommunicationOpOutputDynamicRes(const mindspore::AnfNodePtr &kernel) {
  MS_EXCEPTION_IF_NULL(kernel);
  // The reference count of communication kernel output is not 0.
  if (communication_op_output_ref_count_ != 0) {
    MS_LOG(ERROR) << "The reference count of communication kernel output is not 0.";
    return;
  }

  size_t total = 0;
  std::vector<std::pair<mindspore::device::DeviceAddress *, size_t>> addr_size;
  auto kernel_mod = AnfAlgo::GetKernelMod(kernel);
  MS_EXCEPTION_IF_NULL(kernel_mod);
  auto output_sizes = kernel_mod->GetOutputSizeList();
  for (size_t i = 0; i < output_sizes.size(); ++i) {
    auto device_address = AnfAlgo::GetMutableOutputAddr(kernel, i);
    MS_EXCEPTION_IF_NULL(device_address);
    // The outputs of communication kernel are not released.
    if ((i == 0) && (device_address->ptr_ != nullptr)) {
      MS_LOG(ERROR) << "The outputs of communication kernel are not released.";
      return;
    }
    total += output_sizes[i];
    addr_size.emplace_back(device_address.get(), output_sizes[i]);
  }

  auto device_mem_ptr = AllocTensorMemDynamic(total);
  MS_EXCEPTION_IF_NULL(device_mem_ptr);
  for (const auto &iter : addr_size) {
    MS_EXCEPTION_IF_NULL(iter.first);
    iter.first->set_ptr(device_mem_ptr);
    communication_op_output_ref_count_++;
    device_mem_ptr = AddressOffset(device_mem_ptr, iter.second);
  }
}

void GPUKernelRuntime::FreeKernelDynamicRes(const mindspore::AnfNodePtr &kernel,
                                            const AddressPtrList &kernel_workspaces) {
  MS_EXCEPTION_IF_NULL(kernel);
  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // Free the input of kernel by reference count.
  for (size_t i = 0; i < AnfAlgo::GetInputTensorNum(kernel); ++i) {
    auto kernel_ref_count_ptr = mem_reuse_util_ptr_->GetKernelInputRef(cnode, i);
    if (kernel_ref_count_ptr == nullptr) {
      continue;
    }
    kernel_ref_count_ptr->ref_count_dynamic_use_--;
    if (kernel_ref_count_ptr->ref_count_dynamic_use_ == 0) {
      // Reset the reference count.
      kernel_ref_count_ptr->ref_count_dynamic_use_ = kernel_ref_count_ptr->ref_count_;
      bool is_communication_op = false;
      // The inputs and outputs memory of communication kernel are special, so separate processing.
      FreeCommunicationOpDynamicRes(kernel, i, &is_communication_op);
      if (!is_communication_op) {
        auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, i);
        MS_EXCEPTION_IF_NULL(device_address);
        MS_EXCEPTION_IF_NULL(device_address->ptr_);
        FreeTensorMemDynamic(device_address->ptr_);
        device_address->ptr_ = nullptr;
      }
    }
  }

  // Free the workspace of kernel.
  for (size_t i = 0; i < kernel_workspaces.size(); ++i) {
    auto workspace = kernel_workspaces[i];
    if (workspace != nullptr) {
      MS_EXCEPTION_IF_NULL(workspace->addr);
      FreeTensorMemDynamic(workspace->addr);
      workspace->addr = nullptr;
    }
  }
}

void GPUKernelRuntime::FreeCommunicationOpDynamicRes(const mindspore::AnfNodePtr &kernel, size_t input_idx,
                                                     bool *is_communication_op) {
  MS_EXCEPTION_IF_NULL(kernel);
  // The inputs memory of communication kernel is one piece memory, need release together.
  if (AnfAlgo::GetCNodeName(kernel) == kAllReduceOpName) {
    communication_op_input_ref_count_--;
    if (communication_op_input_ref_count_ == 0) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(device_address->ptr_);
      FreeTensorMemDynamic(device_address->ptr_);
      device_address->ptr_ = nullptr;
    }
    *is_communication_op = true;
    return;
  }

  auto cnode = kernel->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (input_idx + 1 >= cnode->inputs().size()) {
    MS_LOG(EXCEPTION) << "Input index " << input_idx << " is larger than input number " << cnode->inputs().size() - 1
                      << ".";
  }
  auto input_node = cnode->input(input_idx + 1);
  auto kernel_input = AnfAlgo::VisitKernel(input_node, 0);
  // The outputs memory of communication kernel is one piece memory, need release together.
  if (AnfAlgo::GetCNodeName(kernel_input.first) == kAllReduceOpName) {
    communication_op_output_ref_count_--;
    if (communication_op_output_ref_count_ == 0) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel_input.first, 0);
      MS_EXCEPTION_IF_NULL(device_address);
      MS_EXCEPTION_IF_NULL(device_address->ptr_);
      FreeTensorMemDynamic(device_address->ptr_);
      device_address->ptr_ = nullptr;
    }
    *is_communication_op = true;
  }
}

void GPUKernelRuntime::MallocOpMemory(const DeviceAddressPtr address, size_t size, int) {
  auto device_ptr = AllocTensorMemDynamic(size);
  MS_EXCEPTION_IF_NULL(device_ptr);
  address->ptr_ = device_ptr;
  address->mem_dynamic_alloc_ = true;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
