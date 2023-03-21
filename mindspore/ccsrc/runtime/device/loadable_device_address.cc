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

#include "runtime/device/loadable_device_address.h"
#include "include/common/debug/common.h"

namespace mindspore {
namespace device {
namespace {
constexpr size_t kFileAlignSize = 512;
}  // namespace

bool LoadableDeviceAddress::Offload(size_t stream_id) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (mem_offloaded_) {
    MS_LOG(WARNING) << "Trying to offload an offloaded AscendDeviceAddress.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(ptr_);
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  offload_ptr_ = device_context->device_res_manager_->AllocateOffloadMemory(size_);
  if (offload_ptr_ == nullptr) {
    MS_LOG(EXCEPTION) << "Alloc host memory for offloading failed, size: " << size_ << ".";
  }
  if (!AsyncDeviceToHost({}, size_, kTypeUnknown, offload_ptr_, stream_id)) {
    return false;
  }
  device_context->device_res_manager_->FreeMemory(ptr_);
  ptr_ = nullptr;
  mem_offloaded_ = true;
  return true;
}

bool LoadableDeviceAddress::Load(size_t stream_id) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (!mem_offloaded_) {
    MS_LOG(DEBUG) << "Trying to load a loaded AscendDeviceAddress.";
    return true;
  }
  MS_EXCEPTION_IF_NULL(offload_ptr_);
  auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  if (ptr_ == nullptr && !device_context->device_res_manager_->AllocateMemory(this)) {
    MS_LOG(EXCEPTION) << "Alloc memory for loading failed, size: " << size_ << ".";
  }
  MS_EXCEPTION_IF_NULL(ptr_);
  if (!AsyncHostToDevice({}, size_, kTypeUnknown, offload_ptr_, stream_id)) {
    return false;
  }
  device_context->device_res_manager_->FreeOffloadMemory(offload_ptr_);
  offload_ptr_ = nullptr;
  mem_offloaded_ = false;
  return true;
}

bool LoadableDeviceAddress::MoveTo(mindspore::device::StorageType dst, bool async, size_t stream_id) {
  bool ret = Wait();
  if (!ret) {
    MS_LOG(WARNING) << "Wait swapping DeviceAddress failed. Status: " << status_;
    return false;
  }
  if (dst == StorageType::kDevice) {
    if (!MoveToDevice(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to device failed.";
      return false;
    }
  } else if (dst == StorageType::kHost) {
    if (!MoveToHost(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to host failed.";
      return false;
    }
  } else if (dst == StorageType::kFile) {
    if (!MoveToFile(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to file failed.";
      return false;
    }
  }
  return true;
}

bool LoadableDeviceAddress::MoveToHost(bool async, size_t stream_id) const {
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  storage_info_.host_ptr_ = swap_manager->AllocHostMemory(GetFileAlignSize());
  MS_EXCEPTION_IF_NULL(storage_info_.host_ptr_);
  if (status_ == DeviceAddressStatus::kInFile) {
    if (!CopyFileToHost(storage_info_.host_ptr_, storage_info_.file_name_, GetFileAlignSize(), async)) {
      MS_LOG(WARNING) << "Copy data from file to host failed.";
      return false;
    }
    if (async) {
      swap_manager->AddSwappingTensor(this);
      status_ = DeviceAddressStatus::kInFileToHost;
    } else {
      if (!swap_manager->DeleteFile(storage_info_.file_name_)) {
        MS_LOG(WARNING) << "Deleting file " << storage_info_.file_name_ << " failed.";
      }
      storage_info_.file_name_ = "";
      status_ = DeviceAddressStatus::kInHost;
    }
  } else {
    if (!MoveDeviceToHost(storage_info_.host_ptr_, ptr_, size_, async, stream_id)) {
      MS_LOG(WARNING) << "Copy data from device to host failed.";
      return false;
    }
    if (async) {
      swap_manager->AddSwappingTensor(this);
      status_ = DeviceAddressStatus::kInDeviceToHost;
    } else {
      swap_manager->FreeDeviceMemory(ptr_);
      ptr_ = nullptr;
      status_ = DeviceAddressStatus::kInHost;
    }
  }
  return true;
}

bool LoadableDeviceAddress::MoveToDevice(bool async, size_t stream_id) const {
  if (status_ == DeviceAddressStatus::kInDevice) {
    return true;
  }
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (status_ == DeviceAddressStatus::kInFile && !MoveToHost(false, stream_id)) {
    return false;
  }
  ptr_ = swap_manager->AllocDeviceMemory(size_);
  MS_EXCEPTION_IF_NULL(ptr_);
  if (!MoveHostToDevice(ptr_, storage_info_.host_ptr_, size_, async, stream_id)) {
    MS_LOG(WARNING) << "Copy data from host to device failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToDevice;
  } else {
    swap_manager->FreeHostMemory(storage_info_.host_ptr_);
    storage_info_.host_ptr_ = nullptr;
    status_ = DeviceAddressStatus::kInDevice;
  }
  return true;
}

bool LoadableDeviceAddress::MoveToFile(bool async, size_t stream_id) const {
  if (status_ == DeviceAddressStatus::kInFile) {
    return true;
  }
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (status_ == DeviceAddressStatus::kInDevice && !MoveToHost(false, stream_id)) {
    return false;
  }
  storage_info_.file_name_ = GetSwapFileName();
  if (!swap_manager->CreateFile(storage_info_.file_name_)) {
    MS_LOG(WARNING) << "Create file for swapping failed.";
    return false;
  }
  if (!CopyHostToFile(storage_info_.file_name_, storage_info_.host_ptr_, GetFileAlignSize(), async)) {
    MS_LOG(WARNING) << "Copy data from host to file failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToFile;
  } else {
    swap_manager->FreeHostMemory(storage_info_.host_ptr_);
    storage_info_.host_ptr_ = nullptr;
    status_ = DeviceAddressStatus::kInFile;
  }
  return true;
}

bool LoadableDeviceAddress::CopyHostToFile(const std::string &dst, const void *src, size_t size, bool async) const {
  MS_EXCEPTION_IF_NULL(src);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  AsyncIOToken token;
  bool ret = swap_manager->HostMemoryToFile(dst, src, size, async, &token);
  if (!ret) {
    MS_LOG(WARNING) << "Write data from ddr to file[" << dst << "] failed.";
    return ret;
  }
  if (async) {
    MS_EXCEPTION_IF_NULL(swap_event_);
    swap_event_->aio_token_ = token;
  }
  return ret;
}

bool LoadableDeviceAddress::CopyFileToHost(void *dst, const std::string &src, size_t size, bool async) const {
  MS_EXCEPTION_IF_NULL(dst);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  AsyncIOToken token;
  bool ret = swap_manager->FileToHostMemory(dst, src, size, async, &token);
  if (!ret) {
    MS_LOG(WARNING) << "Read data from file[" << src << "] to ddr failed.";
    return ret;
  }
  if (async) {
    MS_EXCEPTION_IF_NULL(swap_event_);
    swap_event_->aio_token_ = token;
  }
  return true;
}

size_t LoadableDeviceAddress::GetFileAlignSize() const {
  return (size_ + kFileAlignSize - 1) / kFileAlignSize * kFileAlignSize;
}

std::string LoadableDeviceAddress::GetSwapFileName() const {
  static size_t swap_file_index = 0;
  return std::to_string(device_id()) + "_" + std::to_string(swap_file_index++) + "_" +
         std::to_string(Common::GetTimeStamp());
}

void LoadableDeviceAddress::SetStorageInfo(const mindspore::device::StorageInfo &storage_info) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  storage_info_ = storage_info;
  if (storage_info_.host_ptr_ != nullptr) {
    status_ = DeviceAddressStatus::kInHost;
  } else if (!storage_info_.file_name_.empty()) {
    status_ = DeviceAddressStatus::kInFile;
  } else {
    status_ = DeviceAddressStatus::kInDevice;
  }
}

void LoadableDeviceAddress::HandOver(mindspore::device::DeviceAddress *other) {
  DeviceAddress::HandOver(other);
  auto loadable_device_address = reinterpret_cast<LoadableDeviceAddress *>(other);
  if (loadable_device_address != nullptr) {
    loadable_device_address->SetStorageInfo(storage_info_);
    loadable_device_address->set_status(status_);
    storage_info_.host_ptr_ = nullptr;
    storage_info_.file_name_ = "";
    status_ = DeviceAddressStatus::kInDevice;
  }
}

bool LoadableDeviceAddress::Wait() const {
  if (swap_event_ == nullptr || !swap_event_->NeedWait()) {
    return true;
  }
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (swap_event_->device_event_ != nullptr && swap_event_->device_event_->NeedWait()) {
    swap_event_->device_event_->WaitEvent();
  } else if (swap_event_->aio_token_ != kInvalidAsyncIOToken) {
    if (!swap_manager->WaitAsyncIO(swap_event_->aio_token_)) {
      MS_LOG(WARNING) << "Wait aio failed.";
      return false;
    }
  } else {
    MS_LOG(WARNING) << "Device address is in moving, but no valid swap event can be found.";
  }
  if (status_ == DeviceAddressStatus::kInFileToHost) {
    if (!swap_manager->DeleteFile(storage_info_.file_name_)) {
      MS_LOG(WARNING) << "Deleting file " << storage_info_.file_name_ << " failed.";
    }
    status_ = DeviceAddressStatus::kInHost;
  } else if (status_ == DeviceAddressStatus::kInDeviceToHost) {
    swap_manager->FreeDeviceMemory(ptr_);
    status_ = DeviceAddressStatus::kInHost;
  } else {
    swap_manager->FreeHostMemory(storage_info_.host_ptr_);
    if (status_ == DeviceAddressStatus::kInHostToDevice) {
      status_ = DeviceAddressStatus::kInHost;
    } else {
      status_ = DeviceAddressStatus::kInFile;
    }
  }
  return true;
}

void LoadableDeviceAddress::SetOffloadPtr(void *offload_ptr) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  offload_ptr_ = offload_ptr;
  mem_offloaded_ = (offload_ptr != nullptr);
}

void *LoadableDeviceAddress::GetOffloadPtr() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  return offload_ptr_;
}

// Return whether DeviceAddress has a valid ptr.
bool LoadableDeviceAddress::IsPtrValid() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  return ptr_ != nullptr || offload_ptr_ != nullptr;
}

// Load first if data is offloaded and return the device ptr.
void *LoadableDeviceAddress::GetValidPtr(size_t stream_id) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (mem_offloaded() && !Load(stream_id)) {
    MS_LOG(EXCEPTION) << "Load offloaded memory failed";
  }
  return ptr_;
}
}  // namespace device
}  // namespace mindspore
