/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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
#include <unordered_map>
#include <utility>
#include <mutex>
#include "ir/tensor.h"
#include "ir/dtype.h"
#include "ir/device_sync.h"
#include "utils/shape_utils.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"
#include "include/backend/device_type.h"
#include "kernel/kernel.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUSimpleMemPlan;
class CPUMemoryManager;
class CPUKernelRuntime;
class CPUDeviceContext;
}  // namespace cpu
namespace ascend {
class AscendKernelRuntime;
class AscendRuntimeCore;
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
class SingleOpInferSession;
class RuntimeUtils;
}  // namespace mindspore

namespace mindspore {
namespace device {
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;

struct StorageInfo {
  void *host_ptr_{nullptr};
  std::string file_name_{""};
  bool host_ptr_mutable_{true};
  bool file_name_mutable_{true};
};

enum class StorageType { kDevice, kHost, kFile };

enum class DeviceAddressStatus {
  kInDevice,
  kInHost,
  kInFile,
  kInDeviceToHost,
  kInHostToDevice,
  kInHostToFile,
  kInFileToHost
};

// The flag of device address.
constexpr size_t kDeviceAddressFlagInit = 0;
// Indicates that it is the device address of ref node.
constexpr size_t kDeviceAddressFlagRefNode = 1;
// Indicates that it is the device address of node which has no user.
constexpr size_t kDeviceAddressFlagNotUsed = 2;
// Indicates that it is the device address of node has init arg and do not need device address.
constexpr size_t kDeviceAddressFlagIgnoreDevicePtr = 4;

class DeviceAddress : public mindspore::DeviceSync {
 public:
  explicit DeviceAddress(void *ptr, size_t size) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
  }
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_dtype_id(type_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index)
      : node_index_(node_index) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_dtype_id(type_id);
  }

  explicit DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
    kernel_tensor_->set_device_name(device_name);
    kernel_tensor_->set_device_id(device_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id, const std::string &device_name,
                         uint32_t device_id) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_dtype_id(type_id);
    kernel_tensor_->set_device_name(device_name);
    kernel_tensor_->set_device_id(device_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
      : node_index_(node_index) {
    kernel_tensor_ = std::make_shared<kernel::KernelTensor>();
    kernel_tensor_->set_device_ptr(ptr);
    kernel_tensor_->set_size(size);
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_dtype_id(type_id);
    kernel_tensor_->set_device_name(device_name);
    kernel_tensor_->set_device_id(device_id);
  }

  explicit DeviceAddress(kernel::KernelTensorPtr &kernel_tensor) : kernel_tensor_(kernel_tensor) {}
  virtual ~DeviceAddress() {
    if (!from_mem_pool_ && deleter_ && GetDevicePtr() != nullptr) {
      deleter_(static_cast<uint8_t *>(GetDevicePtr()));
    }
    SetDevicePtr(nullptr);
  }

  // Asynchronously copy host memory to device side.
  virtual bool AsyncHostToDevice(const ShapeVector &, size_t, TypeId, const void *, size_t) const { return true; }
  // Asynchronously copy device memory to host side.
  virtual bool AsyncDeviceToHost(const ShapeVector &, size_t, TypeId, void *, size_t) const { return true; }
  // Synchronously copy device memory to device side.
  virtual bool SyncDeviceToDevice(const DeviceSync *) const { return true; }
  virtual bool SyncDeviceToDevice(const ShapeVector &, size_t, TypeId, const void *, const std::string &) const {
    return true;
  }
  // Asynchronously copy device memory to device side.
  virtual bool AsyncDeviceToDevice(const ShapeVector &, size_t, TypeId, const void *, const std::string &) const {
    return true;
  }

  // Get kernel tensor pointer.
  const kernel::KernelTensorPtr &kernel_tensor() const { return kernel_tensor_; }
  // Set kernel tensor pointer.
  void set_kernel_tensor(const kernel::KernelTensorPtr &kernel_tensor) { kernel_tensor_ = kernel_tensor; }

  const void *GetPtr() const {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return GetDevicePtr();
  }
  void set_ptr(void *ptr) {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    kernel_tensor_->set_device_ptr(ptr);
    if (ptr != nullptr) {
      const auto &storage_info = GetStorageInfo();
      if (storage_info.host_ptr_ == nullptr && storage_info.file_name_.empty()) {
        status_ = DeviceAddressStatus::kInDevice;
      }
    }
  }
  size_t GetSize() const { return size(); }
  void SetSize(size_t size) { kernel_tensor_->set_size(size); }

  std::string format() const { return kernel_tensor_->GetStringFormat(); }
  void set_format(const std::string &format) { kernel_tensor_->SetStringFormat(format); }
  const std::string &padding_type() const { return kernel_tensor_->padding_type(); }
  void set_padding_type(const std::string &padding_type) { kernel_tensor_->set_padding_type(padding_type); }
  TypeId type_id() const { return kernel_tensor_->dtype_id(); }
  bool from_mem_pool() const { return from_mem_pool_; }
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }
  bool is_ptr_persisted() const { return is_ptr_persisted_; }
  void set_is_ptr_persisted(bool is_ptr_persisted) { is_ptr_persisted_ = is_ptr_persisted; }
  void set_host_shape(const ShapeVector &shape) { host_shape_ = shape; }
  void set_type_id(TypeId type_id) { kernel_tensor_->set_dtype_id(type_id); }
  ShapeVector host_shape() const { return host_shape_; }
  void set_device_shape(const ShapeVector &shape) { device_shape_ = shape; }
  const ShapeVector &device_shape() const { return device_shape_; }
  bool from_persistent_mem() const { return from_persistent_mem_; }
  void set_from_persistent_mem(bool from_persistent_mem) { from_persistent_mem_ = from_persistent_mem; }
  bool need_recycle() const { return need_recycle_; }
  void set_need_recycle(bool need_recycle) { need_recycle_ = need_recycle; }
  virtual bool mem_offloaded() const { return false; }
  void set_status(DeviceAddressStatus status) { status_ = status; }
  DeviceAddressStatus status() const { return status_; }
  virtual DeviceType GetDeviceType() const { return DeviceType::kUnknown; }
  void *GetMutablePtr() const override {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return GetDevicePtr();
  }

  const std::string &device_name() const { return kernel_tensor_->device_name(); }
  uint32_t device_id() const { return kernel_tensor_->device_id(); }

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node) { (void)held_by_nodes_.emplace_back(value_node); }
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const { return held_by_nodes_; }
  void ClearHeldByNodes() { held_by_nodes_.clear(); }

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
      (void)++dynamic_ref_count_;
      MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << dynamic_ref_count_
                    << " for ptr:" << GetDevicePtr();
    }
  }
  void DecreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ <= 0) {
      MS_LOG(EXCEPTION) << "The dynamic reference count is invalid value:" << dynamic_ref_count_;
    }
    (void)--dynamic_ref_count_;
    MS_LOG(DEBUG) << op_object << " decreases dynamic ref count to:" << dynamic_ref_count_
                  << " for ptr:" << GetDevicePtr();
  }

  virtual bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                             TypeId host_type, bool trans_flag) const {
    return true;
  }
#ifdef ENABLE_DEBUGGER
  virtual bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                             const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                             uint32_t root_graph_id, bool force_update, bool trans_flag) const {
    return true;
  }
#endif

  // Return whether DeviceAddress has a valid ptr.
  virtual bool IsPtrValid() const {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return GetDevicePtr() != nullptr;
  }

  // Return the valid device ptr.
  virtual void *GetValidPtr(size_t) {
    auto device_ptr = GetDevicePtr();
    if (user_data() == nullptr || sync_user_data_handler_ == nullptr) {
      return GetDevicePtr();
    }
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    if (sync_user_data_handler_ == nullptr) {
      return device_ptr;
    }
    if (sync_user_data_handler_ != nullptr) {
      sync_user_data_handler_(this);
    }
    sync_user_data_handler_ = nullptr;
    return device_ptr;
  }

  // Offload data from device to host and free device memory
  virtual bool Offload(size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Load data from host to device and free host memory
  virtual bool Load(size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Move data to destination hardware and free resource on source hardware
  virtual bool MoveTo(StorageType, bool, size_t) { MS_LOG(EXCEPTION) << "Not implemented."; }

  virtual bool Wait() const { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Set host ptr data offloaded to
  virtual void SetOffloadPtr(void *) {}

  // Get offloaded host ptr
  virtual void *GetOffloadPtr() const { return nullptr; }

  virtual void SetStorageInfo(const StorageInfo &) {}
  virtual StorageInfo GetStorageInfo() const { return StorageInfo(); }

  virtual void Swap(DeviceAddress *other) {
    MS_EXCEPTION_IF_NULL(other);
    if (other == this) {
      return;
    }
    other->SetDevicePtr(GetDevicePtr());
    SetDevicePtr(nullptr);

    other->from_mem_pool_ = from_mem_pool_;
    other->set_deleter(deleter());
    other->set_sync_user_data_handler(sync_user_data_handler_);
    SetDevicePtr(nullptr);
    from_mem_pool_ = false;
    deleter_ = nullptr;
  }

  virtual void set_swappable(bool) {}
  virtual bool swappable() { return false; }

  // Get user data maintained by the DeviceAddress.
  const UserDataPtr &user_data() const override { return kernel_tensor_->user_data(); }

  // Set user data to the DeviceAddress.
  void set_user_data(const UserDataPtr &user_data) override { kernel_tensor_->set_user_data(user_data); }

  // Free the ptr in user data when the ref count is 0.
  virtual void ClearUserData() {}

  // The interface of flag.
  size_t flag() const { return flag_; }
  void set_flag(size_t flag) { flag_ = flag; }
  void UpdateFlag(size_t flag) { SET_FLAG(flag_, flag); }
  void ClearFlag(size_t flag) { CLEAR_FLAG(flag_, flag); }

  std::pair<AnfNodeWeakPtr, size_t> node_index() const { return node_index_; }
  void set_deleter(const std::function<void(uint8_t *)> &deleter) { deleter_ = deleter; }
  std::function<void(uint8_t *)> deleter() const { return deleter_; }

  using SyncUserDataHandler = void (*)(DeviceAddress *const device_address);
  // For output of pyexecute kernel, the input data is stored in user data and the handler is used to sync data from
  // user data to device ptr.
  SyncUserDataHandler sync_user_data_handler() { return sync_user_data_handler_; }
  void set_sync_user_data_handler(SyncUserDataHandler handler) { sync_user_data_handler_ = handler; }

 protected:
  kernel::KernelTensorPtr kernel_tensor_;
  size_t size() const { return kernel_tensor_->size(); }

  void *GetDevicePtr() const { return kernel_tensor_->device_ptr(); }
  void SetDevicePtr(void *ptr) const { kernel_tensor_->set_device_ptr(ptr); }

  void SetTypeId(TypeId type) const { return kernel_tensor_->set_dtype_id(type); }

  mutable bool from_mem_pool_{false};
  uint8_t *communication_ptr_{nullptr};
  ShapeVector host_shape_{};
  ShapeVector device_shape_{};
  // {node, out_index}
  std::pair<AnfNodeWeakPtr, size_t> node_index_{AnfNodePtr(nullptr), 0};
  // The DeviceAddress is held by ValueNodes. These ValueNodes are outputs of forward network.
  // We need to release the device memory when the reference count of the device address in bprop graph is 0.
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes_;
  // The device address of the node that owns the device address cannot be updated and replaced.
  // Application scenario: set to true when the hardware execution mode requires that ptr cannot be changed during
  // execution.
  bool is_ptr_persisted_{false};
  // Thread lock for ptr_.
  mutable std::recursive_mutex ptr_mutex_;

  // The device address generated in the control flow scene uses dynamic_ref_count_.
  std::atomic_int32_t dynamic_ref_count_{INT32_MAX};

  bool from_persistent_mem_{false};
  bool need_recycle_{false};

  // The device address flag.
  size_t flag_{0};

  // The flag identify where data is stored
  mutable DeviceAddressStatus status_{DeviceAddressStatus::kInDevice};
  // Handler for sync data from user data.
  SyncUserDataHandler sync_user_data_handler_{nullptr};
  // The specified deleter to release memory
  std::function<void(uint8_t *)> deleter_;
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
  friend class mindspore::device::ascend::AscendRuntimeCore;
  friend class mindspore::device::ascend::AscendMemoryManager;
  friend class mindspore::device::ascend::AscendDeviceContext;
#ifndef ENABLE_SECURITY
  friend class mindspore::device::ascend::DataDumper;
#endif
  friend class mindspore::SingleOpInferSession;
  friend class mindspore::RuntimeUtils;
};

using DeviceAddressPtr = std::shared_ptr<DeviceAddress>;
using DeviceAddressPtrList = std::vector<DeviceAddressPtr>;
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_DEVICE_TENSOR_H
