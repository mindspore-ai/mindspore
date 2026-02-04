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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_BASE_H
#define MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_BASE_H

#include <string>
#include <memory>
#include <unordered_map>
#include "device_address/device_address.h"

namespace mindspore {
class StorageBase {
 public:
  using StorageBasePtr = std::shared_ptr<StorageBase>;
  StorageBase() = default;
  explicit StorageBase(device::DeviceAddressPtr &device_data, TypeId type_id)
      : device_data_(device_data), type_id_(type_id) {}
  explicit StorageBase(const StorageBase &storage_base, TypeId type_id)
      : device_data_(storage_base.device_data_), type_id_(type_id) {}
  ~StorageBase();

  uintptr_t DataPtr() const;
  void InplaceReSize(int64_t size);
  int64_t NBytes() const;
  void InplaceCopy(const StorageBasePtr &src, bool non_blocking);
  std::string device() const;
  TypeId GetTypeId() const;
  uint32_t GetStreamId() const;
  const DevicePointerPtr &GetDevicePointer() const;
  const DeviceAddressPtr &GetDeviceAddress() const;
  const MapAllocatorPtr &GetMapAllocator() const;
  void SetDevicePointer(const DevicePointerPtr device_pointer);

 private:
  device::DeviceAddressPtr device_data_{nullptr};
  TypeId type_id_;
};
using StorageBasePtr = std::shared_ptr<StorageBase>;
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_BASE_H
