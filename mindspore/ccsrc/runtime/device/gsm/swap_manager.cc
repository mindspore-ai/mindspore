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

#include "runtime/device/gsm/swap_manager.h"

#include <string>
#include <utility>

namespace mindspore {
namespace device {
SwapManager::SwapManager(size_t stream_id, mindspore::device::DynamicMemPoolBestFit *device_memory_pool,
                         HostMemPoolPtr host_mem_pool)
    : stream_id_(stream_id), device_memory_pool_(device_memory_pool), host_memory_pool_(std::move(host_mem_pool)) {
  io_handle_ = std::make_shared<IOHandle>();
}

template <class Input, class Output>
bool SwapManager::TryAllocate(std::queue<const DeviceAddress *> queue, const Input &input,
                              Output (SwapManager::*allocate_func)(const Input &),
                              const std::function<bool(Output)> &success, Output *output) {
  MS_EXCEPTION_IF_NULL(allocate_func);
  MS_EXCEPTION_IF_NULL(output);
  (*output) = (this->*allocate_func)(input);
  if (success(*output)) {
    return true;
  }
  // Wait swapping tensors.
  while (!queue.empty()) {
    const auto &front = queue.front();
    MS_EXCEPTION_IF_NULL(front);
    if (front->Wait()) {
      (*output) = (this->*allocate_func)(input);
      if (success(*output)) {
        return true;
      }
    }
    queue.pop();
  }
  return false;
}

void *SwapManager::AllocDeviceMemorySimply(const size_t &size) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  return device_memory_pool_->AllocTensorMem(size);
}

void *SwapManager::AllocDeviceMemory(size_t size) {
  void *ret = nullptr;
  void *(SwapManager::*allocate_func)(const size_t &) = &SwapManager::AllocDeviceMemorySimply;
  std::function<bool(void *)> success = [](void *ptr) { return ptr != nullptr; };
  std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
  if (!TryAllocate(swapping_tensors_device_, size, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate device memory failed, size: " << size;
  }
  return ret;
}

std::vector<void *> SwapManager::AllocDeviceContinuousMemSimply(const std::vector<size_t> &size_list) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  return device_memory_pool_->AllocContinuousTensorMem(size_list);
}

std::vector<void *> SwapManager::AllocDeviceContinuousMem(const std::vector<size_t> &size_list) {
  std::vector<void *> ret;
  std::vector<void *> (SwapManager::*allocate_func)(const std::vector<size_t> &) =
    &SwapManager::AllocDeviceContinuousMemSimply;
  std::function<bool(std::vector<void *>)> success = [](const std::vector<void *> &ptrs) { return !ptrs.empty(); };
  std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
  if (!TryAllocate(swapping_tensors_device_, size_list, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate continuous device mem failed, size list: " << size_list;
  }
  return ret;
}

void SwapManager::FreeDeviceMemory(void *ptr) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  device_memory_pool_->FreeTensorMem(ptr);
}

void *SwapManager::AllocHostMemorySimply(const size_t &size) {
  MS_EXCEPTION_IF_NULL(host_memory_pool_);
  return host_memory_pool_->Malloc(size);
}

void *SwapManager::AllocHostMemory(size_t size) {
  void *ret = nullptr;
  void *(SwapManager::*allocate_func)(const size_t &) = &SwapManager::AllocHostMemorySimply;
  std::function<bool(void *)> success = [](void *ptr) { return ptr != nullptr; };
  std::lock_guard<std::mutex> lock(swapping_tensors_host_mutex_);
  if (!TryAllocate(swapping_tensors_host_, size, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate host memory failed, size: " << size;
  }
  return ret;
}

void SwapManager::FreeHostMemory(const void *ptr) {
  MS_EXCEPTION_IF_NULL(host_memory_pool_);
  host_memory_pool_->Free(ptr);
}

bool SwapManager::CreateFile(const std::string &file_name) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  return io_handle_->CreateSwapFile(file_name);
}

bool SwapManager::DeleteFile(const std::string &file_name) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  const auto &iter = file_size_.find(file_name);
  if (iter == file_size_.end()) {
    MS_LOG(WARNING) << "Can not file size for file[" << file_name << "]";
  } else {
    current_used_file_size_ -= iter->second;
    iter->second = 0;
  }
  return io_handle_->DeleteSwapFile(file_name);
}

bool SwapManager::FileToHostMemory(void *host_memory, const std::string &file_name, size_t byte_num, bool async,
                                   AsyncIOToken *sync_key) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  if (async) {
    return io_handle_->ReadAsync(file_name, host_memory, byte_num, sync_key);
  } else {
    return io_handle_->Read(file_name, host_memory, byte_num);
  }
}

bool SwapManager::EnoughFileSpace(const size_t &size) { return current_used_file_size_ + size <= max_file_size_; }

bool SwapManager::HostMemoryToFile(const std::string &file_name, const void *data, size_t byte_num, bool async,
                                   AsyncIOToken *sync_key) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  bool (SwapManager::*allocate_func)(const size_t &size) = &SwapManager::EnoughFileSpace;
  std::function<bool(bool)> success = [](bool ret) { return ret; };
  {
    std::lock_guard<std::mutex> lock(swapping_tensors_file_mutex_);
    bool enough = false;
    if (!TryAllocate(swapping_tensors_file_, byte_num, allocate_func, success, &enough)) {
      MS_LOG(WARNING) << "There is no enough disk space for file writing, size: " << byte_num;
      return false;
    }
  }
  current_used_file_size_ += byte_num;
  file_size_[file_name] = byte_num;
  if (async) {
    return io_handle_->WriteAsync(file_name, data, byte_num, sync_key);
  } else {
    return io_handle_->Write(file_name, data, byte_num);
  }
}

bool SwapManager::WaitAsyncIO(mindspore::device::AsyncIOToken sync_token) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  return io_handle_->Wait(sync_token);
}

void SwapManager::AddSwappableTensor(const DeviceAddressPtr &device_address) {
  swappable_tensors_.push(device_address);
}

void SwapManager::AddSwappingTensor(const mindspore::device::DeviceAddress *device_address) {
  if (device_address == nullptr) {
    return;
  }
  if (device_address->status() == DeviceAddressStatus::kInFileToHost) {
    std::lock_guard<std::mutex> lock(swapping_tensors_file_mutex_);
    swapping_tensors_file_.push(device_address);
  } else if (device_address->status() == DeviceAddressStatus::kInDeviceToHost) {
    std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
    swapping_tensors_device_.push(device_address);
  } else {
    std::lock_guard<std::mutex> lock(swapping_tensors_host_mutex_);
    swapping_tensors_host_.push(device_address);
  }
}
}  // namespace device
}  // namespace mindspore
