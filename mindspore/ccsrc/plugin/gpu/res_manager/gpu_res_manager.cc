/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "plugin/gpu/res_manager/gpu_res_manager.h"
#include <libgen.h>
#include <cuda.h>
#include <utility>
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>
#include "plugin/gpu/res_manager/gpu_device_manager.h"
#include "plugin/gpu/res_manager/event_manager/gpu_event.h"
#include "plugin/gpu/res_manager/mem_manager/gpu_pin_mem_pool.h"
#include "plugin/gpu/res_manager/device_context_conf/op_precision_conf.h"
#include "plugin/gpu/res_manager/device_context_conf/op_tuning_conf.h"
#include "include/runtime/hardware_abstract/collective/collective_manager.h"
#include "include/runtime/hardware_abstract/data_queue/data_queue_mgr.h"
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "ir/tensor_new.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "runtime/hardware_abstract/utils.h"
#include "utils/dlopen_macro.h"
#include "include/runtime/hardware_abstract/collective/collective_comm_lib_loader.h"
#ifdef ENABLE_MPI
#include "plugin/cpu/res_manager/collective/ms_collective_comm_lib.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
std::string GetCurrentDir() {
#ifndef _WIN32
  Dl_info dl_info;
  if (dladdr(reinterpret_cast<void *>(GetCurrentDir), &dl_info) == 0) {
    MS_LOG(WARNING) << "Get dladdr error";
    return "";
  }
  std::string cur_so_path = dl_info.dli_fname;
  return dirname(cur_so_path.data());
#else
  return "";
#endif
}

void GPUResManager::Initialize() {
  uint32_t device_id;
  // Set device id
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    device_id = distributed::collective::CollectiveManager::instance()->local_rank_id();

    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    ms_context->set_param_inner<uint32_t>(MS_CTX_DEVICE_ID, device_id);
  }

  MS_LOG(INFO) << "Set GPU device id index " << device_id;
  // Set device id and initialize device resource.
  if (!InitDevice()) {
    MS_LOG(EXCEPTION) << "GPU InitDevice failed.";
  }

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto device_addr = GPUMemoryAllocator::GetInstance().AllocTensorMem(1, false, false, kDefaultStreamIndex);
  if (!device_addr) {
    MS_LOG(EXCEPTION) << "Dynamic memory pool init error.";
  }
  GPUMemoryAllocator::GetInstance().FreeTensorMem(device_addr);

  swap_manager_ = std::make_shared<SwapManager>(GPUDeviceManager::GetInstance().default_stream_id(),
                                                &GPUMemoryAllocator::GetInstance(), &GPUPinMemPool::GetInstance());

  // Initialize NCCL.
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
#if defined(_WIN32)
    MS_LOG(EXCEPTION) << "windows not support nccl.";
#endif
  }
}

namespace {
float GetCudaCap(const uint32_t device_id) {
  cudaDeviceProp prop;
  (void)cudaGetDeviceProperties(&prop, device_id);
  auto major = prop.major;
  auto minor = prop.minor;
  return static_cast<float>(major * BASE + minor) / BASE;
}
}  // namespace

bool GPUResManager::InitDevice() {
  if (GPUDeviceManager::GetInstance().device_count() <= 0) {
    MS_LOG(ERROR) << "No GPU device found.";
    return false;
  }

  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (!GPUDeviceManager::GetInstance().is_device_id_init()) {
    if (!GPUDeviceManager::GetInstance().set_cur_device_id(device_id)) {
      MS_LOG(ERROR) << "Failed to set current device id: " << SizeToInt(device_id);
      return false;
    }
  }
  // Check the Cuda capability
  const float cuda_cap = GetCudaCap(device_id);
  if (cuda_cap < SUPPORTED_CAP) {
    MS_LOG(WARNING) << "The device with Cuda compute capability " << cuda_cap
                    << " is lower than the minimum required capability " << SUPPORTED_CAP
                    << ", this may cause some unexpected problems and severely affect the results. "
                    << "Eg: the outputs are all zeros.\n"
                    << "Device with a compute capability > " << SUPPORTED_CAP << " is required, "
                    << "and it is recommended to use devices with a compute capability >= " << RECOMMEND_SM;
  }

  // Initialize device resource, such as stream, cudnn and cublas handle.
  GPUDeviceManager::GetInstance().InitDevice();
  return true;
}

void GPUResManager::Destroy() {
  (void)DestroyAllEvents();
  if (DataQueueMgr::GetInstance().IsInit()) {
    if (!DataQueueMgr::GetInstance().IsClosed() && !DataQueueMgr::GetInstance().CloseNotify()) {
      MS_LOG(ERROR) << "Could not close gpu data queue.";
    }
    DataQueueMgr::GetInstance().Release();
  }

  // Release stream, cudnn and cublas handle, etc.
  GPUDeviceManager::GetInstance().ReleaseDevice();

  // Release device memory
  GPUMemoryAllocator::GetInstance().ReleaseDeviceRes();
}

void *GPUResManager::AllocateMemory(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id) {
  if (!BindDeviceToCurrentThread(false)) {
    return nullptr;
  }
  return GPUMemoryAllocator::GetInstance().AllocTensorMem(size, from_persistent_mem, need_recycle, stream_id);
}

void *GPUResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    return nullptr;
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceMemory(size, stream_id);
  }
  return GPUMemoryAllocator::GetInstance().AllocTensorMem(size, false, false, stream_id);
}

void GPUResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr);
}

void GPUResManager::FreePartMemorys(const std::vector<void *> &free_addrs, const std::vector<void *> &keep_addrs,
                                    const std::vector<size_t> &keep_addr_sizes) const {
  GPUMemoryAllocator::GetInstance().FreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

bool GPUResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  std::shared_ptr<AddressAllocator> allocator = address->allocator();
  void *device_ptr = nullptr;

  if (MS_UNLIKELY(allocator != nullptr)) {
    device_ptr = allocator->Alloc(address->GetSize(), stream_id);
  } else {
    auto device_name_in_address = GetDeviceNameByType(static_cast<const DeviceType>(address->GetDeviceType()));
    if (device_name_in_address != "GPU") {
      MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:" << device_name_in_address
                        << ", type name in context: GPU";
    }

    if (address->GetPtr() != nullptr) {
      MS_LOG(ERROR) << "Memory leak detected in device address:" << address->ToString();
      return false;
    }

    if (!BindDeviceToCurrentThread(false)) {
      return false;
    }

    if (stream_id == UINT32_MAX) {
      stream_id = address->stream_id();
    }

    if (swap_manager_ != nullptr) {
      device_ptr = swap_manager_->AllocDeviceMemory(address->GetSize(), stream_id);
    } else {
      device_ptr = GPUMemoryAllocator::GetInstance().AllocTensorMem(address->GetSize(), address->from_persistent_mem(),
                                                                    false, stream_id);
    }
  }

  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  return true;
}

std::vector<void *> GPUResManager::AllocateContinuousMemory(const std::vector<size_t> &size_list,
                                                            uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    std::vector<void *> ptr_list;
    return ptr_list;
  }

  // Memory allocation ensures memory alignment.
  std::vector<size_t> align_size_list;
  for (size_t size : size_list) {
    auto align_size = GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    (void)align_size_list.emplace_back(align_size);
  }
  if (swap_manager_ != nullptr) {
    return swap_manager_->AllocDeviceContinuousMem(align_size_list, stream_id);
  }
  return GPUMemoryAllocator::GetInstance().AllocContinuousTensorMem(align_size_list, stream_id);
}

std::pair<std::vector<size_t>, std::vector<size_t>> GPUResManager::AllocDeviceMemoryForTensorList(
  const std::vector<tensor::TensorPtr> &tensor_list, bool enable_mem_align) {
  std::vector<size_t> before_padding_sizes = GetUniqueTensorListSize(tensor_list);
  std::vector<size_t> after_padding_sizes;
  for (auto &size : before_padding_sizes) {
    auto align_size = GPUMemoryAllocator::GetInstance().AlignMemorySize(size);
    after_padding_sizes.emplace_back(align_size);
  }
  auto stream_id = DefaultStream();
  auto device_ptr_list = AllocateContinuousMemory(before_padding_sizes, stream_id);
  for (size_t i = 0; i < after_padding_sizes.size(); ++i) {
    auto ret = cudaMemset(device_ptr_list[i], 0, after_padding_sizes[i]);
    if (ret != cudaSuccess) {
      MS_LOG(EXCEPTION) << "cudaMemcpy failed, ret[" << static_cast<int>(ret) << "], " << cudaGetErrorString(ret);
    }
    MS_LOG(DEBUG) << "Clear ptr:" << device_ptr_list[i] << ", size:" << after_padding_sizes[i];
  }

  // create device for all tensor in tensor list
  for (size_t i = 0; i < tensor_list.size(); ++i) {
    const auto &tensor = tensor_list[i];
    const auto &ptr = device_ptr_list[i];
    auto device_address =
      std::make_shared<device::DeviceAddress>(ptr, before_padding_sizes[i], device::DeviceType::kGPU, stream_id);
    MS_LOG(DEBUG) << "Create DeviceAddress, ptr:" << ptr << ", size:" << before_padding_sizes[i]
                  << ", shape:" << tensor->shape() << ", data_type:" << TypeIdToString(tensor->data_type());
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(tensor->device_address());
    device::DeviceContextKey host_key = {device_address->GetDeviceType(), device_address->device_id()};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    host_context->device_res_manager_->SyncAllStreams();
    DeviceAddressExtPtr src_ext = std::make_shared<DeviceAddressExt>(tensor->format(), tensor->data_type(),
                                                                     tensor->shape(), tensor->storage_info());
    DeviceAddressExtPtr dst_ext =
      std::make_shared<DeviceAddressExt>(Format::DEFAULT_FORMAT, tensor->data_type(), tensor->shape());
    SyncCopy(device_address, tensor->device_address(), device_address->stream_id(), src_ext, dst_ext);
    tensor->set_device_address(device_address);
    tensor->set_format(Format::DEFAULT_FORMAT);
    tensor->set_storage_info(nullptr);
  }
  return std::make_pair(before_padding_sizes, after_padding_sizes);
}

tensor::TensorPtr GPUResManager::GetSliceByTensorListIndexHandle(const std::vector<tensor::TensorPtr> &tensor_list,
                                                                 const std::vector<size_t> &before_padding_size,
                                                                 const std::vector<size_t> &after_padding_size,
                                                                 size_t start, size_t end) {
  if (start >= tensor_list.size() || end > tensor_list.size()) {
    MS_EXCEPTION(ValueError) << "start:" << start << ", end:" << end << ", but tensor_list size:" << tensor_list.size();
  }
  size_t size = std::accumulate(after_padding_size.begin() + start, after_padding_size.begin() + end - 1,
                                before_padding_size[end - 1]);
  ShapeVector shape = {static_cast<int64_t>(size / UnitSizeInBytes(tensor_list[start]->data_type()))};
  auto tensor = tensor::from_spec(tensor_list[start]->data_type(), shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(tensor_list[start]->device_address());
  auto ptr = tensor_list[start]->device_address()->GetMutablePtr();

  auto stream_id = DefaultStream();
  auto device_address = std::make_shared<device::DeviceAddress>(ptr, size, device::DeviceType::kGPU, stream_id);
  tensor->set_device_address(device_address);
  tensor->set_storage_info(nullptr);
  return tensor;
}

tensor::TensorPtr GPUResManager::GetSliceByPaddingShapeHandle(const tensor::TensorPtr &first_tensor, size_t start,
                                                              size_t end) {
  auto type_id = first_tensor->data_type();
  auto type_size = UnitSizeInBytes(type_id);
  size_t tensor_size = (end - start) * type_size;
  ShapeVector shape = {static_cast<int64_t>(end - start)};
  auto tensor = tensor::from_spec(type_id, shape, device::DeviceType::kNone);
  MS_EXCEPTION_IF_NULL(first_tensor->device_address());
  auto ptr = first_tensor->device_address()->GetMutablePtr();
  auto offset_size = start * type_size;

  auto stream_id = DefaultStream();
  auto device_address = std::make_shared<device::DeviceAddress>(reinterpret_cast<uint8_t *>(ptr) + offset_size,
                                                                tensor_size, device::DeviceType::kGPU, stream_id);
  MS_LOG(DEBUG) << "Create DeviceAddress, offset size to ptr0:" << offset_size << ", tensor_size:" << tensor_size
                << ", shape:" << shape << ", data_type:" << TypeIdToString(type_id);
  tensor->set_device_address(device_address);
  tensor->set_storage_info(nullptr);
  return tensor;
}

// Relevant function to manage memory statistics
size_t GPUResManager::GetTotalMemStatistics() const { return GPUMemoryAllocator::GetInstance().TotalMemStatistics(); }

size_t GPUResManager::GetTotalUsedMemStatistics() const {
  return GPUMemoryAllocator::GetInstance().TotalUsedMemStatistics();
}

size_t GPUResManager::GetTotalIdleMemStatistics() const {
  return GPUMemoryAllocator::GetInstance().TotalIdleMemStatistics();
}
size_t GPUResManager::GetTotalEagerFreeMemStatistics() const {
  return GPUMemoryAllocator::GetInstance().TotalEagerFreeMemStatistics();
}

size_t GPUResManager::GetUsedMemPeakStatistics() const {
  return GPUMemoryAllocator::GetInstance().MaxMemAllocatedStatistics();
}

size_t GPUResManager::GetReservedMemPeakStatistics() const {
  return GPUMemoryAllocator::GetInstance().MaxMemReservedStatistics();
}

std::unordered_map<std::string, std::size_t> GPUResManager::GetBlockCountsStatistics() const {
  return GPUMemoryAllocator::GetInstance().BlockCountsStatistics();
}

std::unordered_map<std::string, std::size_t> GPUResManager::GetBlockUnitSizeStatistics() const {
  return GPUMemoryAllocator::GetInstance().BlockUnitSizeStatistics();
}

DeviceMemInfo GPUResManager::GetCommonMemBlocksInfoStatistics() const {
  return GPUMemoryAllocator::GetInstance().CommonMemBlocksInfoStatistics();
}

DeviceMemInfo GPUResManager::GetPersistentMemBlocksInfoStatistics() const {
  return GPUMemoryAllocator::GetInstance().PersistentMemBlocksInfoStatistics();
}

void GPUResManager::ResetMaxMemoryReserved() { GPUMemoryAllocator::GetInstance().ResetMaxMemReserved(); }

void GPUResManager::ResetMaxMemoryAllocated() { GPUMemoryAllocator::GetInstance().ResetMaxMemAllocated(); }

bool GPUResManager::SyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                             size_t stream_id, const DeviceAddressExtPtr &src_ext,
                             const DeviceAddressExtPtr &dst_ext) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == DeviceType::kGPU && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return SyncHostToDevice(dst_device_sync, src_device_sync, stream_id);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kGPU) {
    return SyncDeviceToHost(dst_device_sync, src_device_sync, stream_id);
  }
  return SyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id);
}

bool GPUResManager::AsyncCopy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                              size_t stream_id, bool keep_src, const DeviceAddressExtPtr &src_ext,
                              const DeviceAddressExtPtr &dst_ext) const {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == DeviceType::kGPU && src_device_sync->GetDeviceType() == DeviceType::kCPU) {
    return AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id);
  }
  if (dst_device_sync->GetDeviceType() == DeviceType::kCPU && src_device_sync->GetDeviceType() == DeviceType::kGPU) {
    return AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id);
  }
  return AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id);
}

bool GPUResManager::Copy(void *dst, const void *src, uint64_t size, CopyType kind, size_t stream_id) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get stream id:" << stream_id;
  }
  switch (kind) {
    case CopyType::kD2H: {
      if (!CudaDriver::CopyDeviceMemToHostAsync(dst, src, size, stream)) {
        MS_LOG(ERROR) << "CopyDeviceMemToHostAsync failed from:" << dst << " to:" << src << " size:" << size;
      }
      break;
    }
    case CopyType::kH2D: {
      if (!CudaDriver::CopyHostMemToDeviceAsync(dst, src, size, stream)) {
        MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed from:" << dst << " to:" << src << " size:" << size;
        return false;
      }
      break;
    }
    case CopyType::kD2D: {
      if (!CudaDriver::CopyDeviceMemToDeviceAsync(dst, src, size, stream)) {
        MS_LOG(ERROR) << "CopyDeviceMemToDeviceAsync failed from:" << dst << " to:" << src << " size:" << size;
      }
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Invalid copy type:" << kind;
  }
  return true;
}

bool GPUResManager::CopyDirectly(void *dst, size_t dst_size, const void *src, size_t src_size, CopyType kind) const {
  MS_EXCEPTION_IF_NULL(dst);
  MS_EXCEPTION_IF_NULL(src);
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(dst, const_cast<void *>(src),
                                                             src_size > dst_size ? dst_size : src_size);
}

bool GPUResManager::SyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                     size_t stream_id) const {
  if (!AsyncDeviceToHost(dst_device_sync, src_device_sync, stream_id)) {
    return false;
  }
  return SyncStream(stream_id);
}

bool GPUResManager::SyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                     size_t stream_id) const {
  if (!AsyncHostToDevice(dst_device_sync, src_device_sync, stream_id)) {
    return false;
  }
  return SyncStream(stream_id);
}

bool GPUResManager::SyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                       size_t stream_id) const {
  if (!AsyncDeviceToDevice(dst_device_sync, src_device_sync, stream_id)) {
    return false;
  }
  return SyncStream(stream_id);
}

bool GPUResManager::AsyncDeviceToHost(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                      size_t stream_id) const {
  return Copy(dst_device_sync, src_device_sync, stream_id, cudaMemcpyKind::cudaMemcpyDeviceToHost);
}

bool GPUResManager::AsyncHostToDevice(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                                      size_t stream_id) const {
  return Copy(dst_device_sync, src_device_sync, stream_id, cudaMemcpyKind::cudaMemcpyHostToDevice);
}

bool GPUResManager::AsyncDeviceToDevice(const DeviceAddressPtr &dst_device_sync,
                                        const DeviceAddressPtr &src_device_sync, size_t stream_id) const {
  return Copy(dst_device_sync, src_device_sync, stream_id, cudaMemcpyKind::cudaMemcpyDeviceToDevice);
}

bool GPUResManager::Copy(const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync,
                         size_t stream_id, cudaMemcpyKind copy_type) const {
  const auto &dst_device_address = dynamic_cast<const DeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const DeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  MS_LOG(DEBUG) << "Async copy from device address:" << src_device_address->ToString()
                << " to:" << dst_device_address->ToString();
  if (dst_device_address->GetDevicePtr() == src_device_address->GetDevicePtr()) {
    MS_LOG(DEBUG) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  // The input or output may be empty.
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(DEBUG) << "No need copy for size 0, from device address:" << src_device_address->ToString()
                  << " to:" << dst_device_address->ToString();
    return true;
  }
  size_t copy_size = src_device_address->GetSize();
  if (dst_device_address->GetSize() < src_device_address->GetSize()) {
    MS_LOG(WARNING) << "Src size is greater than dst size, src device address:" << src_device_address->ToString()
                    << " dst:" << dst_device_address->ToString();
    copy_size = dst_device_address->GetSize();
  }

  auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  if (stream == nullptr) {
    MS_LOG(EXCEPTION) << "Failed to get stream id:" << stream_id
                      << " for src device address:" << src_device_address->ToString();
  }
  MS_EXCEPTION_IF_NULL(src_device_address->GetDevicePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetDevicePtr());
  switch (copy_type) {
    case cudaMemcpyKind::cudaMemcpyDeviceToHost: {
      if (!CudaDriver::CopyDeviceMemToHostAsync(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                                                copy_size, stream)) {
        MS_LOG(ERROR) << "CopyDeviceMemToHostAsync failed from device address:" << src_device_address->ToString()
                      << " to:" << dst_device_address->ToString();
      }
      break;
    }
    case cudaMemcpyKind::cudaMemcpyHostToDevice: {
      if (!CudaDriver::CopyHostMemToDeviceAsync(dst_device_address->GetDevicePtr(), src_device_address->GetDevicePtr(),
                                                copy_size, stream)) {
        MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed from device address:" << src_device_address->ToString()
                      << " to:" << dst_device_address->ToString();
        return false;
      }
      break;
    }
    case cudaMemcpyKind::cudaMemcpyDeviceToDevice: {
      if (!CudaDriver::CopyDeviceMemToDeviceAsync(dst_device_address->GetDevicePtr(),
                                                  src_device_address->GetDevicePtr(), copy_size, stream)) {
        MS_LOG(ERROR) << "CopyDeviceMemToHostAsync failed from device address:" << src_device_address->ToString()
                      << " to:" << dst_device_address->ToString();
      }
      break;
    }
    default:
      MS_LOG(EXCEPTION) << "Invalid copy type for src device address:" << src_device_address
                        << " dst:" << dst_device_address;
  }
  return true;
}

bool GPUResManager::CreateStream(size_t *stream_id) const {
  return GPUDeviceManager::GetInstance().CreateStream(stream_id);
}

bool GPUResManager::CreateStreamWithPriority(size_t *stream_id, int32_t priority) const {
  return GPUDeviceManager::GetInstance().CreateStreamWithPriority(stream_id, priority);
}

size_t GPUResManager::QueryStreamSize() const { return GPUDeviceManager::GetInstance().QueryStreamSize(); }

std::vector<uint32_t> GPUResManager::GetStreamIds() const { return GPUDeviceManager::GetInstance().GetStreamIds(); }

bool GPUResManager::single_op_multi_stream_enable() const {
  return GPUDeviceManager::GetInstance().single_op_multi_stream_enable();
}

void GPUResManager::set_single_op_multi_stream_enable(bool single_op_multi_stream_enable) {
  return GPUDeviceManager::GetInstance().set_single_op_multi_stream_enable(single_op_multi_stream_enable);
}

void *GPUResManager::GetStream(size_t stream_id) const { return GPUDeviceManager::GetInstance().GetStream(stream_id); }

size_t GPUResManager::GetCommunicationStreamID() const {
  MS_LOG(WARNING) << "CommunicationStreamID is no create yet, return default stream.";
  return GPUDeviceManager::GetInstance().default_stream_id();
}

bool GPUResManager::DestroyStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().DestroyStream(stream_id);
}

void GPUResManager::SetCurrentStreamId(size_t stream_id) {
  GPUDeviceManager::GetInstance().set_current_stream(stream_id);
}

size_t GPUResManager::GetCurrentStreamId() const { return GPUDeviceManager::GetInstance().current_stream(); }

bool GPUResManager::QueryStream(size_t stream_id) const {
  return GPUDeviceManager::GetInstance().QueryStream(stream_id);
}

bool GPUResManager::SyncStream(size_t stream_id) const { return GPUDeviceManager::GetInstance().SyncStream(stream_id); }

bool GPUResManager::SyncAllStreams(bool) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Fail to bind device to current thread";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncAllStreams();
}
bool GPUResManager::SyncNotDefaultStreams() const { return GPUDeviceManager::GetInstance().SyncNotDefaultStreams(); }

size_t GPUResManager::DefaultStream() const { return GPUDeviceManager::GetInstance().default_stream_id(); }

// cudaEventRecordDefault 0x0 | cudaEventRecordExternal 0x1 | cudaEventWaitExternal 0x1, no need to set again.
DeviceEventPtr GPUResManager::CreateRuntimeEvent(bool enable_blocking, bool enable_record_wait) {
  if (!enable_blocking && !enable_record_wait) {
    MS_LOG(INTERNAL_EXCEPTION) << "Bad parameters, enable_blocking is false and enable_record_wait is false.";
  }
  uint32_t flag = cudaEventDefault;
  flag |= cudaEventDisableTiming;
  flag |= cudaEventBlockingSync;
  return std::make_shared<GpuEvent>(flag);
}

DeviceEventPtr GPUResManager::CreateEventWithFlag(bool enable_timing, bool blocking, bool) {
  uint32_t flag =
    (blocking ? cudaEventBlockingSync : cudaEventDefault) | (enable_timing ? cudaEventDefault : cudaEventDisableTiming);
  auto event = std::make_shared<GpuEvent>(flag);
  MS_EXCEPTION_IF_NULL(event);
  std::lock_guard<std::mutex> lock(device_events_mutex_);
  device_events_.push_back(event);
  return event;
}

bool GPUResManager::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (!event->DestroyEvent()) {
    MS_LOG(ERROR) << "DestroyEvent failed.";
    return false;
  }

  std::lock_guard<std::mutex> lock(device_events_mutex_);
  const auto &iter = std::find(device_events_.begin(), device_events_.end(), event);
  if (iter == device_events_.end()) {
    MS_LOG(ERROR) << "Can't find specified device event.";
    return false;
  }
  (void)device_events_.erase(iter);
  return true;
}

bool GPUResManager::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    // Reduce the scopt to prevent deadlock.
    std::lock_guard<std::mutex> lock(device_events_mutex_);
    device_events_inner = device_events_;
    device_events_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    MS_EXCEPTION_IF_NULL(event);
    if (!event->DestroyEvent()) {
      MS_LOG(ERROR) << "DestroyEvent failed.";
    }
  });
  device_events_.clear();
  return true;
}

bool GPUResManager::LoadCollectiveCommLib() {
#ifdef ENABLE_MPI
  std::string nvidia_comm_lib_name = GetCurrentDir() + "/gpu" + std::to_string(CUDA_VERSION / 1000) + "." +
                                     std::to_string(CUDA_VERSION / 10 % 10) + "/libnvidia_collective.so";
  auto loader = std::make_shared<CollectiveCommLibLoader>(nvidia_comm_lib_name);
  MS_EXCEPTION_IF_NULL(loader);
  if (!loader->Initialize()) {
    MS_LOG(EXCEPTION) << "Loading NCCL collective library failed.";
    return false;
  }
  void *collective_comm_lib_handle = loader->collective_comm_lib_ptr();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_handle);

  auto instance_func = DlsymFuncObj(communication_lib_instance, collective_comm_lib_handle);
  collective_comm_lib_ = instance_func();
  MS_EXCEPTION_IF_NULL(collective_comm_lib_);
  return true;
#else
  return false;
#endif
}

static thread_local bool cur_thread_device_inited{false};

bool GPUResManager::BindDeviceToCurrentThread(bool force_bind) const {
  if (cur_thread_device_inited && !force_bind) {
    return true;
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  if (!CudaDriver::SetDevice(UintToInt(device_id))) {
    MS_LOG(ERROR) << "Failed to set device id: " << device_id;
    return false;
  }

  cur_thread_device_inited = true;
  return true;
}

std::shared_ptr<void> GPUResManager::AllocateHostMemory(size_t size) const {
  void *ptr;
  if (CudaDriver::AllocHostPinnedMem(size, &ptr) != size) {
    MS_LOG(ERROR) << "Failed to allow host pinned memory.";
    return nullptr;
  }

  return std::shared_ptr<void>(ptr, [](void *ptr) -> void {
    if (ptr != nullptr) {
      CudaDriver::FreeHostPinnedMem(ptr);
    }
  });
}

DynamicMemPool *GPUResManager::GetMemoryPool() {
  if (MS_UNLIKELY(memory_pool_ == nullptr)) {
    memory_pool_ = &GPUMemoryAllocator::GetInstance();
  }
  return memory_pool_;
}

MS_REGISTER_HAL_COPY_FUNC(
  DeviceType::kGPU,
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kGPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->SyncCopy(dst_device_sync, src_device_sync, stream_id, src_ext, dst_ext);
  }),
  ([](const DeviceAddressPtr &dst_device_sync, const DeviceAddressPtr &src_device_sync, size_t stream_id, bool keep_src,
      const DeviceAddressExtPtr &src_ext, const DeviceAddressExtPtr &dst_ext) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kGPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->AsyncCopy(dst_device_sync, src_device_sync, stream_id, keep_src, src_ext,
                                                        dst_ext);
  }),
  ([](void *dst, const void *src, uint64_t size, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::DeviceContextKey host_key = {DeviceType::kGPU, device_id};
    device::DeviceContext *host_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(host_key);
    MS_EXCEPTION_IF_NULL(host_context);
    MS_EXCEPTION_IF_NULL(host_context->device_res_manager_);
    return host_context->device_res_manager_->Copy(dst, src, size, device::CopyType::kD2H, stream_id);
  }));

REGISTER_DEVICE_PTR_DELETER_MAKER(device::DeviceType::kGPU, ([](void *ptr, bool from_mem_pool) {
                                    if (ptr != nullptr && from_mem_pool) {
                                      GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr);
                                    }
                                  }));
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
