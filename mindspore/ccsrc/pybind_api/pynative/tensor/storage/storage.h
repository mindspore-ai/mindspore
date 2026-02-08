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

#ifndef MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_H
#define MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_H

#include <Python.h>
#include "pybind_api/pynative/tensor/storage/storage_base.h"
#include <string>
#include <memory>
#include <unordered_map>

namespace mindspore {
class Storage {
 public:
  Storage() = default;
  explicit Storage(const StorageBasePtr &storage_base) : storage_base_(storage_base) {}
  ~Storage();

  uintptr_t DataPtr() const;
  void InplaceReSize(int64_t size);
  int64_t NBytes() const;
  void InplaceCopy(const Storage &src, bool non_blocking = false);
  std::string device() const;
  const StorageBasePtr get_storage_base() const { return storage_base_; }
  StorageBasePtr get_mutable_storage_base() const { return storage_base_; }
  TypeId GetTypeId() const;
  uint32_t GetStreamId() const;
  const DevicePointerPtr &GetDevicePointer() const;
  const DeviceAddressPtr &GetDeviceAddress() const;
  const MapAllocatorPtr &GetMapAllocator() const;
  void SetDevicePointer(const DevicePointerPtr device_pointer);

 private:
  StorageBasePtr storage_base_;
};

}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PYBIND_API_IR_TENSOR_STORAGE_STORAGE_H
