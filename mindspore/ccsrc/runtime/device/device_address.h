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
class AscendDeviceContext;
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

  explicit DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id)
      : ptr_(ptr), size_(size), device_name_(device_name), device_id_(device_id) {}
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id, const std::string &device_name,
                         uint32_t device_id)
      : ptr_(ptr), size_(size), format_(format), type_id_(type_id), device_name_(device_name), device_id_(device_id) {}
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
      : ptr_(ptr),
        size_(size),
        format_(format),
        type_id_(type_id),
        node_index_(node_index),
        device_name_(device_name),
        device_id_(device_id) {}
  virtual ~DeviceAddress() { ptr_ = nullptr; }

  const void *GetPtr() const { return ptr_; }
  void set_ptr(void *ptr) { ptr_ = ptr; }
  size_t GetSize() const { return size_; }
  void SetSize(size_t size) { size_ = size; }

  std::string format() const { return format_; }
  TypeId type_id() const { return type_id_; }
  bool from_mem_pool() const { return from_mem_pool_; }
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }
  bool is_ptr_persisted() const { return is_ptr_persisted_; }
  void set_is_ptr_persisted(bool is_ptr_persisted) { is_ptr_persisted_ = is_ptr_persisted; }
  void set_host_shape(const ShapeVector &shape) { host_shape_ = shape; }
  ShapeVector host_shape() const { return host_shape_; }
  bool from_persistent_mem() const { return from_persistent_mem_; }
  void set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }
  virtual void set_status(DeviceAddressStatus status) {}
  virtual DeviceAddressStatus status() const { return DeviceAddressStatus::kInDevice; }
  virtual DeviceAddressType DeviceType() const { return DeviceAddressType::kUnknown; }
  void *GetMutablePtr() const override { return ptr_; }
  std::string device_name() const { return device_name_; }
  uint32_t device_id() const { return device_id_; }

  virtual void SetNodeIndex(const AnfNodePtr &node, size_t out_index) { node_index_ = {node, out_index}; }
  KernelWithIndex GetNodeIndex() const {
    return node_index_.first.expired() ? KernelWithIndex{nullptr, node_index_.second}
                                       : KernelWithIndex{node_index_.first.lock(), node_index_.second};
  }

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_conut) { dynamic_ref_count_ = dynamic_ref_conut; }
  int32_t dynamic_ref_count() const { return dynamic_ref_count_; }
  void IncreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ < INT32_MAX) {
      dynamic_ref_count_++;
      MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << dynamic_ref_count_ << " for ptr:" << ptr_;
    }
  }
  void DecreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ <= 0) {
      MS_LOG(EXCEPTION) << "The dynamic reference count is invalid value:" << dynamic_ref_count_;
    }
    dynamic_ref_count_--;
    MS_LOG(DEBUG) << op_object << " decreases dynamic ref count to:" << dynamic_ref_count_ << " for ptr:" << ptr_;
  }

  virtual bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                             TypeId host_type, bool trans_flag) const {
    return true;
  }
#ifdef ENABLE_DEBUGGER
  virtual bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                             const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                             uint32_t root_graph_id = 0) const {
    return true;
  }
#endif

 protected:
  const void *ptr() const { return ptr_; }
  size_t size() const { return size_; }

  mutable void *ptr_{nullptr};
  size_t size_{0};
  string format_{"DefaultFormat"};
  TypeId type_id_{kNumberTypeFloat16};
  mutable bool from_mem_pool_{false};
  uint8_t *communication_ptr_{nullptr};
  ShapeVector host_shape_{};
  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  // The device address of the node that owns the device address cannot be updated and replaced.
  // Application scenario: set to true when the hardware execution mode requires that ptr cannot be changed during
  // execution.
  bool is_ptr_persisted_{false};

  // The device address generated in the control flow scene uses dynamic_ref_count_.
  std::atomic_int32_t dynamic_ref_count_{INT32_MAX};

  // The key of device context.
  std::string device_name_{""};
  uint32_t device_id_{0};
  bool from_persistent_mem_{false};

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
  friend class mindspore::device::ascend::AscendDeviceContext;
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
