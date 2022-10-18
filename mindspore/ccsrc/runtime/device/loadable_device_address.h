/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LOADABLE_DEVICE_ADDRESS_H_
#define MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LOADABLE_DEVICE_ADDRESS_H_

#include <string>
#include "runtime/device/device_address.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
// LoadableDeviceAddress provide the ability to offload data on device to host and load it back later.
class LoadableDeviceAddress : public DeviceAddress {
 public:
  LoadableDeviceAddress(void *ptr, size_t size) : DeviceAddress(ptr, size) {}
  LoadableDeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : DeviceAddress(ptr, size, format, type_id) {}
  LoadableDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                        const KernelWithIndex &node_index)
      : DeviceAddress(ptr, size, format, type_id, node_index) {}
  LoadableDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                        const std::string &device_name, uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, device_name, device_id) {}
  LoadableDeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id)
      : DeviceAddress(ptr, size, device_name, device_id) {}
  LoadableDeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                        const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
      : DeviceAddress(ptr, size, format, type_id, node_index, device_name, device_id) {}

 protected:
  DeviceContext *GetDeviceContext() const {
    return DeviceContextManager::GetInstance().GetOrCreateDeviceContext({device_name_, device_id_});
  }

 public:
  // Offload data from device to host and free device memory
  bool Offload(size_t stream_id) override {
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

  // Load data from host to device and free host memory
  bool Load(size_t stream_id) override {
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

  // Set host ptr data offloaded to
  void SetOffloadPtr(void *offload_ptr) override {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    offload_ptr_ = offload_ptr;
    mem_offloaded_ = (offload_ptr != nullptr);
  }

  // Get offloaded host ptr
  void *GetOffloadPtr() const override {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return offload_ptr_;
  }
};
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CCSRC_RUNTIME_DEVICE_LOADABLE_DEVICE_ADDRESS_H_
