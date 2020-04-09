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

#ifndef MINDSPORE_DEVICE_TENSOR_H
#define MINDSPORE_DEVICE_TENSOR_H

#include <string>
#include <vector>
#include <memory>
#include "ir/dtype.h"

using std::string;

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan;
class CPUResourceManager;
class CPUKernelRuntime;
}  // namespace cpu
namespace ascend {
class AscendKernelRuntime;
class AscendMemoryManager;
namespace tasksink {
class TaskGenerator;
}  // namespace tasksink
}  // namespace ascend
namespace gpu {
class GPUKernelRuntime;
class GPUMemoryManager;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

namespace mindspore {
namespace device {
class DeviceAddress {
 public:
  explicit DeviceAddress(void *ptr, size_t size) : ptr_(ptr), size_(size) {}
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : ptr_(ptr), size_(size), format_(format), type_id_(type_id) {}
  virtual ~DeviceAddress() { ptr_ = nullptr; }
  virtual bool SyncDeviceToHost(const std::vector<int> &shape, size_t size, TypeId type, void *host_ptr) const = 0;
  virtual bool SyncHostToDevice(const std::vector<int> &shape, size_t size, TypeId type,
                                const void *host_ptr) const = 0;
  const void *GetPtr() const { return ptr_; }
  size_t GetSize() const { return size_; }
  std::string format() const { return format_; }
  TypeId type_id() const { return type_id_; }

 protected:
  const void *ptr() const { return ptr_; }
  size_t size() const { return size_; }
  void set_ptr(void *ptr) { ptr_ = ptr; }
  void *ptr_{nullptr};
  size_t size_{0};
  size_t ref_count_{0};
  string format_{"DefaultFormat"};
  TypeId type_id_{kNumberTypeFloat16};
  bool mem_dynamic_alloc_{false};
  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUResourceManager;
  friend class mindspore::device::cpu::CPUKernelRuntime;
  friend class mindspore::device::gpu::GPUKernelRuntime;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::ascend::AscendKernelRuntime;
  friend class mindspore::device::ascend::AscendMemoryManager;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
