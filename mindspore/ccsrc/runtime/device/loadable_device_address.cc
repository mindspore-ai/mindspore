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
#include "include/common/utils/offload_context.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace {
constexpr size_t kFileAlignSize = 512;
constexpr char kSwapFileSuffix[] = ".data";
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
  if (status_ == DeviceAddressStatus::kInDevice && ptr_ == nullptr) {
    MS_LOG(INFO) << "Skip move empty device address.";
    return true;
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
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (storage_info_.host_ptr_ == nullptr || storage_info_.host_ptr_mutable_) {
    storage_info_.host_ptr_ = swap_manager->AllocHostMemory(GetFileAlignSize());
    if (storage_info_.host_ptr_ == nullptr) {
      MS_LOG(WARNING) << "Allocating host memory failed, size: " << size_;
      return false;
    }
  }
  if (status_ == DeviceAddressStatus::kInFile) {
    if (!CopyFileToHost(storage_info_.host_ptr_, storage_info_.file_name_, size_, async)) {
      MS_LOG(WARNING) << "Copy data from file to host failed.";
      return false;
    }
    if (async) {
      swap_manager->AddSwappingTensor(this);
      status_ = DeviceAddressStatus::kInFileToHost;
    } else {
      if (storage_info_.file_name_mutable_) {
        (void)swap_manager->DeleteFile(storage_info_.file_name_);
        storage_info_.file_name_ = "";
      }
      status_ = DeviceAddressStatus::kInHost;
    }
  } else {
    if (!CopyDeviceToHost(storage_info_.host_ptr_, ptr_, size_, async, stream_id)) {
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
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (status_ == DeviceAddressStatus::kInFile) {
#if defined(RT_MEMORY_P2PDMA)
    if (ptr_ == nullptr) {
      ptr_ = swap_manager->AllocDeviceMemory(size_);
    }
    MS_EXCEPTION_IF_NULL(ptr_);
    if (FileToDeviceDirectly(ptr_, size_, storage_info_.file_name_, stream_id)) {
      if (storage_info_.file_name_mutable_ && !storage_info_.file_name.empty()) {
        (void)swap_manager->DeleteFile(storage_info_.file_name);
        storage_info_.file_name = "";
      }
      if (storage_info_.host_ptr_mutable_) {
        swap_manager->FreeHostMemory(storage_info_.host_ptr_);
        storage_info_.host_ptr_ = nullptr;
      }
      status_ = DeviceAddressStatus::kInDevice;
      return true;
    }
#endif
    if (!MoveToHost(false, stream_id)) {
      return false;
    }
  }
  if (ptr_ == nullptr) {
    ptr_ = swap_manager->AllocDeviceMemory(size_);
    if (ptr_ == nullptr) {
      MS_LOG(WARNING) << "Allocating device memory failed, size: " << size_;
      return false;
    }
  }
  if (!CopyHostToDevice(ptr_, storage_info_.host_ptr_, size_, async, stream_id)) {
    MS_LOG(WARNING) << "Copy data from host to device failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToDevice;
  } else {
    if (storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(storage_info_.host_ptr_);
      storage_info_.host_ptr_ = nullptr;
    }

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
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (status_ == DeviceAddressStatus::kInDevice) {
#if defined(RT_MEMORY_P2PDMA)
    if (storage_info_.file_name_.empty() || storage_info_.file_name_mutable_) {
      storage_info_.file_name_ = GetSwapFileName();
    }
    if (DeviceToFileDirectly(ptr_, size_, storage_info_.file_name_, stream_id)) {
      status_ = DeviceAddressStatus::kInFile;
      if (ptr_ != nullptr) {
        swap_manager->FreeDeviceMemory(ptr_);
        ptr_ = nullptr;
      }
      if (storage_info_.host_ptr_ != nullptr) {
        swap_manager->FreeHostMemory(storage_info_.host_ptr_);
        storage_info_.host_ptr_ = nullptr;
      }
      return true;
    }
#endif
    if (!MoveToHost(false, stream_id)) {
      return false;
    }
  }
  if (storage_info_.file_name_.empty() || storage_info_.file_name_mutable_) {
    storage_info_.file_name_ = GetSwapFileName();
    if (!swap_manager->CreateFile(storage_info_.file_name_, GetFileAlignSize())) {
      MS_LOG(WARNING) << "Create file for swapping failed.";
      return false;
    }
  }
  if (!CopyHostToFile(storage_info_.file_name_, storage_info_.host_ptr_, size_, async)) {
    MS_LOG(WARNING) << "Copy data from host to file failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToFile;
  } else {
    if (storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(storage_info_.host_ptr_);
      storage_info_.host_ptr_ = nullptr;
    }
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
    swap_event_.aio_token_ = token;
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
    swap_event_.aio_token_ = token;
  }
  return true;
}

void LoadableDeviceAddress::ReleaseResource() {
  if (status_ == DeviceAddressStatus::kInDevice) {
    return;
  }

  const bool need_delete_file = !storage_info_.file_name_.empty() && storage_info_.file_name_mutable_;
  const bool need_free_host = storage_info_.host_ptr_ != nullptr && storage_info_.host_ptr_mutable_;
  if (need_delete_file || need_free_host) {
    auto device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(device_context);
    const auto swap_manager = device_context->device_res_manager_->swap_manager();
    MS_EXCEPTION_IF_NULL(swap_manager);
    if (need_delete_file) {
      (void)swap_manager->DeleteFile(storage_info_.file_name_);
    }
    if (need_free_host) {
      swap_manager->FreeHostMemory(storage_info_.host_ptr_);
    }
  }
}

size_t LoadableDeviceAddress::GetFileAlignSize() const {
  return (size_ + kFileAlignSize - 1) / kFileAlignSize * kFileAlignSize;
}

std::string LoadableDeviceAddress::GetSwapFileName() const {
  static size_t swap_file_index = 0;
  std::string file_dir;
  const auto &offload_context = OffloadContext::GetInstance();
  if (offload_context != nullptr) {
    const auto real_dir = FileUtils::GetRealPath(offload_context->offload_path().c_str());
    if (!real_dir.has_value()) {
      MS_LOG(EXCEPTION) << "Invalid offload path[" << offload_context->offload_path()
                        << "]. Please check offload_path configuration.";
    }
    file_dir = real_dir.value() + "/";
  }
  return file_dir + std::to_string(device_id()) + "_" + std::to_string(swap_file_index++) + "_" +
         std::to_string(Common::GetTimeStamp()) + kSwapFileSuffix;
}

void LoadableDeviceAddress::SetStorageInfo(const mindspore::device::StorageInfo &storage_info) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  storage_info_ = storage_info;
  if (storage_info_.host_ptr_ != nullptr) {
    status_ = DeviceAddressStatus::kInHost;
    storage_info_.host_ptr_mutable_ = false;
  } else if (!storage_info_.file_name_.empty()) {
    status_ = DeviceAddressStatus::kInFile;
    storage_info_.file_name_mutable_ = false;
  } else {
    status_ = DeviceAddressStatus::kInDevice;
  }
}

StorageInfo LoadableDeviceAddress::GetStorageInfo() const { return storage_info_; }

void LoadableDeviceAddress::Swap(mindspore::device::DeviceAddress *other) {
  DeviceAddress::Swap(other);
  if (other == this) {
    return;
  }
  auto loadable_device_address = reinterpret_cast<LoadableDeviceAddress *>(other);
  if (loadable_device_address != nullptr) {
    loadable_device_address->storage_info_ = storage_info_;
    loadable_device_address->status_ = status_;
    loadable_device_address->offload_ptr_ = offload_ptr_;
    loadable_device_address->mem_offloaded_ = mem_offloaded_;
    storage_info_.host_ptr_ = nullptr;
    storage_info_.file_name_ = "";
    storage_info_.host_ptr_mutable_ = true;
    storage_info_.file_name_mutable_ = true;
    status_ = DeviceAddressStatus::kInDevice;
    offload_ptr_ = nullptr;
    mem_offloaded_ = false;
  }
}

bool LoadableDeviceAddress::Wait() const {
  if (!swap_event_.NeedWait()) {
    return true;
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (swap_event_.device_event_ != nullptr && swap_event_.device_event_->NeedWait()) {
    swap_event_.device_event_->WaitEvent();
  } else if (swap_event_.aio_token_ != kInvalidAsyncIOToken) {
    if (!swap_manager->WaitAsyncIO(swap_event_.aio_token_)) {
      MS_LOG(WARNING) << "Wait aio failed.";
      return false;
    }
  } else {
    MS_LOG(WARNING) << "Device address is in moving, but no valid swap event can be found.";
  }
  if (status_ == DeviceAddressStatus::kInFileToHost) {
    if (storage_info_.file_name_mutable_) {
      (void)swap_manager->DeleteFile(storage_info_.file_name_);
      storage_info_.file_name_ = "";
    }
    status_ = DeviceAddressStatus::kInHost;
  } else if (status_ == DeviceAddressStatus::kInDeviceToHost) {
    swap_manager->FreeDeviceMemory(ptr_);
    status_ = DeviceAddressStatus::kInHost;
  } else {
    if (storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(storage_info_.host_ptr_);
      storage_info_.host_ptr_ = nullptr;
    }
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
  return ptr_ != nullptr || offload_ptr_ != nullptr || storage_info_.host_ptr_ != nullptr ||
         !storage_info_.file_name_.empty();
}

// Load first if data is offloaded and return the device ptr.
void *LoadableDeviceAddress::GetValidPtr(size_t stream_id) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (mem_offloaded() && !Load(stream_id)) {
    MS_LOG(EXCEPTION) << "Load offloaded memory failed";
  }
  if (!MoveToDevice(false)) {
    MS_LOG(ERROR) << "Move data to device failed.";
    return nullptr;
  }

  return DeviceAddress::GetValidPtr(stream_id);
}
}  // namespace device
}  // namespace mindspore
