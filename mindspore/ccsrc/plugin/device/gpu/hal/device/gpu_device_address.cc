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

#include "plugin/device/gpu/hal/device/gpu_device_address.h"
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "plugin/device/gpu/hal/device/gpu_device_manager.h"
#include "plugin/device/gpu/hal/device/gpu_memory_allocator.h"
#include "plugin/device/gpu/hal/hardware/gpu_device_context.h"
#include "plugin/device/gpu/hal/device/gpu_hash_table_util.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#ifdef ENABLE_DEBUGGER
#include "debug/debug_services.h"
#include "debug/tensor_load.h"
#include "debug/debugger/debugger.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "include/common/debug/rdr/recorder_manager.h"
#endif

namespace mindspore {
namespace device {
namespace gpu {
bool GPUDeviceAddress::SyncDeviceToHost(size_t size, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (size > size_) {
    MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
    return true;
  }

  MS_EXCEPTION_IF_NULL(host_ptr);
  auto ret = GPUDeviceManager::GetInstance().SyncAllStreams();
  if (!ret) {
#ifdef ENABLE_DUMP_IR
    mindspore::RDR::TriggerAll();
#endif
    MS_LOG(ERROR) << "SyncStream failed";
    return ret;
  }
  if (size != size_) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << size_;
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(offload_ptr_);
    return GPUDeviceManager::GetInstance().CopyHostMemToHost(host_ptr, offload_ptr_, size);
  } else {
    MS_EXCEPTION_IF_NULL(ptr_);
    return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(host_ptr, ptr_, size);
  }
}

bool GPUDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << size_;
    return true;
  }
  if (size > size_) {
    MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << size_;
    return true;
  }
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (size != size_) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << size_;
  }

  // Bind device by device name and device id on the current thread.
  if (device_name_ != "") {
    auto device_context =
      device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
    auto gpu_device_context = dynamic_cast<GPUDeviceContext *>(device_context);
    MS_EXCEPTION_IF_NULL(gpu_device_context);
    if (!gpu_device_context->device_res_manager_->BindDeviceToCurrentThread(false)) {
      MS_LOG(EXCEPTION) << "BindDeviceToCurrentThread failed.";
    }
  }

  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (mem_offloaded()) {
    MS_EXCEPTION_IF_NULL(offload_ptr_);
    return GPUDeviceManager::GetInstance().CopyHostMemToHost(offload_ptr_, host_ptr, size);
  } else {
    MS_EXCEPTION_IF_NULL(ptr_);
    auto &stream = GPUDeviceManager::GetInstance().default_stream();
    MS_EXCEPTION_IF_NULL(stream);
    if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(ptr_, host_ptr, size, stream)) {
      MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
      return false;
    }
    return GPUDeviceManager::GetInstance().SyncStream(stream);
  }
}

bool GPUDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr) const {
  return SyncDeviceToHost(size, host_ptr);
}

namespace {
bool SyncUserDataToDevice(const UserDataPtr &user_data, const void *host_ptr, size_t size) {
  MS_EXCEPTION_IF_NULL(user_data);
  MS_EXCEPTION_IF_NULL(host_ptr);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);

  if (*user_data_type == UserDataType::kUserTypeHashTable) {
#if CUDA_VERSION > 11000 && defined(__linux__)
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = hashtable_func_list.find({*key_type, *value_type});
    if (iter != hashtable_func_list.end()) {
      return std::get<kSyncFuncIndex>(iter->second)(user_data, host_ptr, size);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
#else
    MS_LOG(EXCEPTION) << "Invalid platform or cuda version for gpu hash table.";
#endif
  }
  return true;
}
}  // namespace

bool GPUDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr,
                                        const std::string &format) const {
  if (user_data_ != nullptr) {
    return SyncUserDataToDevice(user_data_, host_ptr, size);
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (execution_mode != kPynativeMode) {
    return SyncHostToDevice(size, host_ptr);
  }

  // PyNative mode need copy async to improve performance.
  MS_EXCEPTION_IF_NULL(host_ptr);
  bool need_sync = (size != 0) && (size_ != 0) && (size <= size_);
  if (!need_sync) {
    return true;
  }
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  return GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(ptr_, host_ptr, size, stream);
}

bool GPUDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_gpu_device = dynamic_cast<const GPUDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_gpu_device);
  if (src_gpu_device->mem_offloaded()) {
    return SyncHostToDevice(src_gpu_device->host_shape(), src_gpu_device->GetSize(), src_gpu_device->type_id(),
                            src_gpu_device->GetOffloadPtr(), src_gpu_device->format());
  } else {
    return SyncDeviceToDevice(src_gpu_device->host_shape(), src_gpu_device->GetSize(), src_gpu_device->type_id(),
                              src_gpu_device->GetPtr(), src_gpu_device->format());
  }
}

bool GPUDeviceAddress::SyncDeviceToDevice(const ShapeVector &, size_t size, TypeId type, const void *src_ptr,
                                          const std::string &format) const {
  MS_LOG(DEBUG) << "SyncDeviceToDevice, dst(format:" << format_ << ", type_id:" << TypeIdLabel(type_id_)
                << ", size:" << size_ << "), src(format:" << format << ", type_id:" << TypeIdLabel(type)
                << ", size:" << size << ")";
  if (ptr_ == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  if (type_id_ > kMonadTypeBegin && type_id_ < kMonadTypeEnd) {
    return true;
  }
  // The input or output may be empty.
  if ((size == 0) || (size_ == 0)) {
    MS_LOG(INFO) << "No need sync, src device size: " << size << ", dst device size: " << size_;
    return true;
  }
  if (size_ < size) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << size << ", dst size is: " << size_;
    return false;
  }
  if (format_ != format || type_id_ != type) {
    MS_LOG(ERROR) << "Format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << format_ << "), type_id:" << TypeIdLabel(type_id_);
    return false;
  }

  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(ptr_);
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  if (mem_offloaded()) {
    if (!GPUDeviceManager::GetInstance().CopyDeviceMemToHostAsync(offload_ptr_, src_ptr, size, stream)) {
      MS_LOG(ERROR) << "CopyDeviceMemToDeviceAsync failed";
      return false;
    }
  } else if (!GPUDeviceManager::GetInstance().CopyDeviceMemToDeviceAsync(ptr_, src_ptr, size, stream)) {
    MS_LOG(ERROR) << "CopyDeviceMemToDeviceAsync failed";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncStream(stream);
}

bool GPUDeviceAddress::AsyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr,
                                         size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MS_ERROR_IF_NULL(ptr_);
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyHostMemToDeviceAsync(ptr_, host_ptr, size, stream),
                              "CopyHostMemToDeviceAsync failed");
  return true;
}

bool GPUDeviceAddress::AsyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr,
                                         size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MS_ERROR_IF_NULL(ptr_);
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyDeviceMemToHostAsync(host_ptr, ptr_, size, stream),
                              "CopyHostMemToDeviceAsync failed");
  return true;
}

void GPUDeviceAddress::ClearDeviceMemory() {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (offload_ptr_ != nullptr) {
    auto device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->device_res_manager_->FreeOffloadMemory(offload_ptr_);
    offload_ptr_ = nullptr;
  }
  if (ptr_ != nullptr && from_mem_pool_) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr_);
    ptr_ = nullptr;
  }
}

void GPUDeviceAddress::ClearUserData() {
  if (user_data_ == nullptr) {
    return;
  }

  auto user_data_type = user_data_->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
#if CUDA_VERSION > 11000 && defined(__linux__)
    auto key_type = user_data_->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data_->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = hashtable_func_list.find({*key_type, *value_type});
    if (iter != hashtable_func_list.end()) {
      return std::get<kClearFuncIndex>(iter->second)(user_data_);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
#else
    MS_LOG(EXCEPTION) << "Invalid platform or cuda version for gpu hash table.";
#endif
  }
}

GPUDeviceAddress::~GPUDeviceAddress() { ClearDeviceMemory(); }

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Load tensor to host and create tensor_data object for the loaded tensor.
 */
#ifdef ENABLE_DEBUGGER
bool GPUDeviceAddress::LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                                     const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                                     uint32_t root_graph_id, bool force_update, bool) const {
  bool ret = false;
  if (size_ == 0) {
    return true;
  }

  MS_EXCEPTION_IF_NULL(Debugger::GetInstance());
  if (Debugger::GetInstance()->TensorExistsInCurrent(tensor_name) && !force_update) {
    MS_LOG(INFO) << tensor_name << " already loaded for this step so not loading it again.";
    return true;
  }

  if (host_type > TypeId::kNumberTypeEnd || host_type < TypeId::kNumberTypeBegin || host_type == kNumberTypeComplex64) {
    MS_LOG(INFO) << "Cannot create tensor with type: " << TypeIdLabel(host_type);
    return false;
  }
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
  size_t host_size = out_tensor->data().nbytes();
  if (host_size == 0) {
    MS_LOG(INFO) << "Host size is 0 for tensor: " << tensor_name << ", no need to load.";
  }
  auto ret_rt_memcpy = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  if (!ret_rt_memcpy) {
    MS_LOG(ERROR) << "Copy device mem to host failed";
    return ret;
  }
  auto tensor_data = std::make_shared<mindspore::TensorData>();
  MS_EXCEPTION_IF_NULL(tensor_data);
  tensor_data->SetName(tensor_name);
  tensor_data->SetExecutionOrder(execution_order);
  tensor_data->SetSlot(slot);
  tensor_data->SetTensor(out_tensor);
  tensor_data->SetDataPtr(static_cast<char *>(out_tensor->data_c()));
  tensor_data->SetByteSize(out_tensor->data().nbytes());
  tensor_data->SetType(host_type);
  tensor_data->SetShape(out_tensor->shape());
  tensor_data->SetRootGraphId(root_graph_id);
  tensor_data->SetFormat(host_fmt);
  ret = Debugger::GetInstance()->LoadNewTensor(tensor_data, keep_prev);
  MS_LOG(INFO) << "E2E tensor name is " << tensor_name;
  return ret;
}
#endif
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
