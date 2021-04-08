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
#include "ir/device_sync.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace device {
class Bucket;
namespace cpu {
class CPUSimpleMemPlan;
class CPUMemoryManager;
class CPUKernelRuntime;
}  // namespace cpu
namespace ascend {
class AscendKernelRuntime;
class AscendMemoryManager;
class DataDumper;
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
enum class DeviceAddressStatus { kInDevice, kInHost, kInDeviceToHost, kInHostToDevice };
enum class DeviceAddressType { kUnknown, kAscend, kCPU, kGPU };

class DeviceAddress : public mindspore::DeviceSync {
 public:
  explicit DeviceAddress(void *ptr, size_t size) : ptr_(ptr), size_(size) {}
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : ptr_(ptr), size_(size), format_(format), type_id_(type_id) {}
  virtual ~DeviceAddress() { ptr_ = nullptr; }
  const void *GetPtr() const { return ptr_; }
  size_t GetSize() const { return size_; }
  std::string format() const { return format_; }
  TypeId type_id() const { return type_id_; }
  void set_host_shape(const ShapeVector &shape) { host_shape_ = shape; }
  virtual void set_status(DeviceAddressStatus status) {}
  virtual DeviceAddressStatus status() const { return DeviceAddressStatus::kInDevice; }
  virtual DeviceAddressType DeviceType() const { return DeviceAddressType::kUnknown; }
  void *GetMutablePtr() const override { return ptr_; }
  virtual bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                             TypeId host_type, bool trans_flag) const {
    return true;
  }
#ifdef ENABLE_DEBUGGER
  virtual bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                             const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev) const {
    return true;
  }
#endif

 protected:
  const void *ptr() const { return ptr_; }
  size_t size() const { return size_; }
  void set_ptr(void *ptr) { ptr_ = ptr; }
  void *ptr_{nullptr};
  size_t size_{0};
  size_t ref_count_{0};
  string format_{"DefaultFormat"};
  TypeId type_id_{kNumberTypeFloat16};
  bool from_mem_pool_{false};
  uint8_t *communication_ptr_{nullptr};
  ShapeVector host_shape_{};
  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUMemoryManager;
  friend class mindspore::device::cpu::CPUKernelRuntime;
  friend class mindspore::device::gpu::GPUKernelRuntime;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::ascend::AscendKernelRuntime;
  friend class mindspore::device::ascend::AscendMemoryManager;
  friend class mindspore::device::ascend::DataDumper;
  friend class mindspore::device::Bucket;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
