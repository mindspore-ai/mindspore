/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <map>
#include <utility>
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
class CPUDeviceContext;
}  // namespace cpu
namespace ascend {
class AscendKernelRuntime;
class AscendMemoryManager;
#ifndef ENABLE_SECURITY
class DataDumper;
#endif
namespace tasksink {
class TaskGenerator;
}  // namespace tasksink
}  // namespace ascend
namespace gpu {
class GPUKernelRuntime;
class GPUMemoryManager;
class GPUDeviceContext;
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

namespace mindspore {
namespace device {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
enum class DeviceAddressStatus { kInDevice, kInHost, kInDeviceToHost, kInHostToDevice };
enum class DeviceAddressType { kUnknown, kAscend, kCPU, kGPU };
static const std::map<DeviceAddressType, std::string> kDeviceTypeToName = {{DeviceAddressType::kUnknown, "Unknown"},
                                                                           {DeviceAddressType::kAscend, "Ascend"},
                                                                           {DeviceAddressType::kCPU, "CPU"},
                                                                           {DeviceAddressType::kGPU, "GPU"}};

class DeviceAddress : public mindspore::DeviceSync {
 public:
  explicit DeviceAddress(void *ptr, size_t size) : ptr_(ptr), size_(size) {}
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id)
      : ptr_(ptr), size_(size), format_(format), type_id_(type_id) {}
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index)
      : ptr_(ptr), size_(size), format_(format), type_id_(type_id), node_index_(node_index) {}
  virtual ~DeviceAddress() { ptr_ = nullptr; }
  const void *GetPtr() const { return ptr_; }
  size_t GetSize() const { return size_; }
  void SetSize(size_t size) { size_ = size; }
  std::string format() const { return format_; }
  TypeId type_id() const { return type_id_; }
  bool from_mem_pool() const { return from_mem_pool_; }
  void set_host_shape(const ShapeVector &shape) { host_shape_ = shape; }
  virtual void set_status(DeviceAddressStatus status) {}
  virtual DeviceAddressStatus status() const { return DeviceAddressStatus::kInDevice; }
  virtual DeviceAddressType DeviceType() const { return DeviceAddressType::kUnknown; }
  void *GetMutablePtr() const override { return ptr_; }
  virtual void SetNodeIndex(const AnfNodePtr &node, size_t out_index) { node_index_ = {node, out_index}; }
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
  KernelWithIndex GetNodeIndex() const {
    return node_index_.first.expired() ? KernelWithIndex{nullptr, node_index_.second}
                                       : KernelWithIndex{node_index_.first.lock(), node_index_.second};
  }
  mutable void *ptr_{nullptr};
  size_t size_{0};
  string format_{"DefaultFormat"};
  TypeId type_id_{kNumberTypeFloat16};
  mutable bool from_mem_pool_{false};
  uint8_t *communication_ptr_{nullptr};
  ShapeVector host_shape_{};
  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  friend class KernelRuntime;
  friend class MemoryManager;
  friend class mindspore::device::ascend::tasksink::TaskGenerator;
  friend class mindspore::device::cpu::CPUSimpleMemPlan;
  friend class mindspore::device::cpu::CPUMemoryManager;
  friend class mindspore::device::cpu::CPUKernelRuntime;
  friend class mindspore::device::cpu::CPUDeviceContext;
  friend class mindspore::device::gpu::GPUKernelRuntime;
  friend class mindspore::device::gpu::GPUMemoryManager;
  friend class mindspore::device::gpu::GPUDeviceContext;
  friend class mindspore::device::ascend::AscendKernelRuntime;
  friend class mindspore::device::ascend::AscendMemoryManager;
#ifndef ENABLE_SECURITY
  friend class mindspore::device::ascend::DataDumper;
#endif
  friend class mindspore::device::Bucket;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
