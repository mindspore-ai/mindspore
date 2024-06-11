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
using kernel::AddressCommon;
using kernel::AddressCommonPtr;
using kernel::KernelTensor;
using kernel::KernelTensorPtr;

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
// Indicates that it is the ptr of device address is nullptr.
constexpr size_t kDeviceAddressFlagNullptr = 8;

class DeviceAddress : public mindspore::DeviceSync {
 public:
  explicit DeviceAddress(const KernelTensorPtr &kernel_tensor)
      : kernel_tensor_(kernel_tensor), address_common_(kernel_tensor_->address_common()) {}

  explicit DeviceAddress(void *ptr, size_t size) {
    address_common_ = std::make_shared<AddressCommon>(ptr, size);
    kernel_tensor_ = std::make_shared<KernelTensor>();
  }
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id) {
    kernel_tensor_ = std::make_shared<KernelTensor>();
    address_common_ = kernel_tensor_->address_common();
    address_common_->pointer_ref_count_->set_ptr(ptr);
    address_common_->size_ = size;
    address_common_->dtype_id_ = type_id;
    kernel_tensor_->SetStringFormat(format);
  }
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index)
      : node_index_(node_index) {
    kernel_tensor_ = std::make_shared<KernelTensor>();
    address_common_ = kernel_tensor_->address_common();
    address_common_->pointer_ref_count_->set_ptr(ptr);
    address_common_->size_ = size;
    address_common_->dtype_id_ = type_id;
    kernel_tensor_->SetStringFormat(format);
  }

  explicit DeviceAddress(void *ptr, size_t size, const std::string &device_name, uint32_t device_id) {
    kernel_tensor_ = std::make_shared<KernelTensor>();
    address_common_ = kernel_tensor_->address_common();
    address_common_->pointer_ref_count_->set_ptr(ptr);
    address_common_->size_ = size;
    address_common_->device_name_ = device_name;
    kernel_tensor_->set_device_id(device_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const string &format, TypeId type_id, const std::string &device_name,
                         uint32_t device_id) {
    kernel_tensor_ = std::make_shared<KernelTensor>();
    address_common_ = kernel_tensor_->address_common();
    address_common_->pointer_ref_count_->set_ptr(ptr);
    address_common_->size_ = size;
    address_common_->device_name_ = device_name;
    address_common_->dtype_id_ = type_id;
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_device_id(device_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId type_id,
                         const std::string &device_name, uint32_t device_id, uint32_t stream_id) {
    address_common_ =
      std::make_shared<AddressCommon>(ptr, size, shape_vector, format, type_id, device_name, device_id, stream_id);
  }
  explicit DeviceAddress(void *ptr, size_t size, const std::string &format, TypeId type_id,
                         const KernelWithIndex &node_index, const std::string &device_name, uint32_t device_id)
      : node_index_(node_index) {
    kernel_tensor_ = std::make_shared<KernelTensor>();
    address_common_ = kernel_tensor_->address_common();
    address_common_->pointer_ref_count_->set_ptr(ptr);
    address_common_->size_ = size;
    address_common_->device_name_ = device_name;
    address_common_->dtype_id_ = type_id;
    kernel_tensor_->SetStringFormat(format);
    kernel_tensor_->set_device_id(device_id);
  }

  virtual ~DeviceAddress() {
    if (!from_mem_pool() && deleter_ && GetDevicePtr() != nullptr) {
      deleter_(static_cast<uint8_t *>(GetDevicePtr()));
      SetDevicePtr(nullptr);
    } else {
      address_common_->pointer_ref_count_ = nullptr;
    }
  }
  virtual bool AsyncHostToDevice(size_t size, TypeId /* type */, const void *host_ptr) const { return true; }

  virtual bool AsyncHostToDevice(size_t size, const void *host_ptr) const { return true; }
  virtual bool AsyncDeviceToHost(size_t size, void *host_ptr) const { return true; }

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
  virtual bool CopyDeviceToHost(void *dst, const void *src, const size_t &size) const { return true; }
  virtual bool CopyHostToDevice(void *dst, const void *src, const size_t &size) const { return true; }
  virtual void DeviceSynchronizerInit() { MS_LOG(EXCEPTION) << "Not implemented."; }

  // Get kernel tensor pointer.
  const KernelTensorPtr &kernel_tensor() const { return kernel_tensor_; }
  void set_kernel_tensor(const KernelTensorPtr &kernel_tensor) {
    kernel_tensor_ = kernel_tensor;
    address_common_ = kernel_tensor_->address_common();
  }

  void set_device_synchronizer(const DeviceSynchronizerPtr &device_synchronizer) {
    MS_EXCEPTION_IF_NULL(kernel_tensor_);
    kernel_tensor_->set_device_synchronizer(device_synchronizer);
  }

  const void *GetPtr() const {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return GetDevicePtr();
  }
  void set_ptr(void *ptr) {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    address_common_->pointer_ref_count_->set_ptr(ptr);
    if (ptr != nullptr) {
      const auto &storage_info = GetStorageInfo();
      if (storage_info.host_ptr_ == nullptr && storage_info.file_name_.empty()) {
        status_ = DeviceAddressStatus::kInDevice;
      }
    }
  }
  size_t GetSize() const { return size(); }
  void SetSize(size_t size) { address_common_->size_ = size; }

  std::string format() const { return kernel::GetFormatFromEnumToStr(address_common_->format_); }
  void set_format(const std::string &format) { address_common_->format_ = kernel::GetFormatFromStrToEnum(format); }
  Format GetFormatEnum() const { return address_common_->format_; }
  const std::string &padding_type() const { return padding_type_; }
  void set_padding_type(const std::string &padding_type) { padding_type_ = padding_type; }
  TypeId type_id() const { return address_common_->dtype_id_; }
  void set_type_id(TypeId type_id) { address_common_->dtype_id_ = type_id; }
  bool from_mem_pool() const { return address_common_->pointer_ref_count_->from_mem_pool(); }
  void set_from_mem_pool(bool from_mem_pool) const {
    address_common_->pointer_ref_count_->set_from_mem_pool(from_mem_pool);
  }
  virtual void set_communication_ptr(uint8_t *communication_ptr) { MS_LOG(EXCEPTION) << "Not implemented error."; }
  bool is_ptr_persisted() const { return is_ptr_persisted_; }
  void set_is_ptr_persisted(bool is_ptr_persisted) { is_ptr_persisted_ = is_ptr_persisted; }
  void set_host_shape(const ShapeVector &shape) { kernel_tensor_->set_host_shape(shape); }
  const ShapeVector &host_shape() const { return kernel_tensor_->host_shape(); }
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
  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const { return address_common_->shape_vector_; }

  const TensorStorageInfoPtr GetTensorStorageInfo() const override {
    if (address_common_ == nullptr) {
      return nullptr;
    }

    return address_common_->tensor_storage_info_;
  }
  void set_tensor_storage_info(const TensorStorageInfoPtr &tensor_storage_info) {
    address_common_->tensor_storage_info_ = tensor_storage_info;
  }

  const std::string &device_name() const { return address_common_->device_name_; }
  uint32_t device_id() const { return address_common_->device_id_; }

  void set_stream_id(uint32_t stream_id) { address_common_->stream_id_ = stream_id; }
  const uint32_t stream_id() const { return address_common_->stream_id_; }

  void AddHeldByNode(const std::weak_ptr<ValueNode> &value_node) { (void)held_by_nodes_.emplace_back(value_node); }
  std::vector<std::weak_ptr<ValueNode>> held_by_nodes() const { return held_by_nodes_; }
  void ClearHeldByNodes() { held_by_nodes_.clear(); }

  virtual void SetNodeIndex(const AnfNodePtr &node, size_t out_index) { node_index_ = {node, out_index}; }
  KernelWithIndex GetNodeIndex() const {
    return node_index_.first.expired() ? KernelWithIndex{nullptr, node_index_.second}
                                       : KernelWithIndex{node_index_.first.lock(), node_index_.second};
  }

  size_t IncreaseCounter() { return address_common_->pointer_ref_count_->IncreaseCounter(); }
  size_t DecreaseCounter() { return address_common_->pointer_ref_count_->DecreaseCounter(); }

  // The related interface of reference count operation.
  void set_original_ref_count(size_t original_ref_count) const override {
    address_common_->pointer_ref_count_->set_original_ref_count(original_ref_count);
  }
  size_t original_ref_count() const override { return address_common_->pointer_ref_count_->original_ref_count(); }
  void set_ref_count(size_t ref_count) const override { address_common_->pointer_ref_count_->set_ref_count(ref_count); }
  size_t ref_count() const override { return address_common_->pointer_ref_count_->ref_count(); }
  void ResetRefCount() override { address_common_->pointer_ref_count_->ResetRefCount(); }

  void IncreaseOriginalRefCount() {
    if (original_ref_count() < SIZE_MAX) {
      address_common_->pointer_ref_count_->IncreaseOriginalRefCount();
    }
  }
  void DecreaseOriginalRefCount() {
    if ((original_ref_count() < SIZE_MAX) && (original_ref_count() > 0)) {
      address_common_->pointer_ref_count_->DecreaseOriginalRefCount();
    }
  }
  size_t DecreaseRefCount() { return address_common_->pointer_ref_count_->DecreaseRefCount(); }

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_count) {
    address_common_->pointer_ref_count_->set_dynamic_ref_count(dynamic_ref_count);
  }

  int32_t dynamic_ref_count() const { return address_common_->pointer_ref_count_->dynamic_ref_count(); }
  void IncreaseDynamicRefCount(const std::string &op_object) {
    address_common_->pointer_ref_count_->IncreaseDynamicRefCount(op_object);
  }
  int32_t DecreaseDynamicRefCount(const std::string &op_object) {
    return address_common_->pointer_ref_count_->DecreaseDynamicRefCount(op_object);
  }

  virtual bool DumpMemToFile(const std::string &filepath, const std::string &host_fmt, const ShapeVector &host_shape,
                             TypeId host_type, bool trans_flag) const {
    return true;
  }
#ifdef ENABLE_DEBUGGER
  virtual bool LoadMemToHost(const std::string &tensor_name, int execution_order, const std::string &host_fmt,
                             const ShapeVector &host_shape, TypeId host_type, size_t slot, bool keep_prev,
                             uint32_t root_graph_id, bool force_update, bool trans_flag, bool async_copy = true) const {
    return true;
  }
#endif

  // Return whether DeviceAddress has a valid ptr.
  virtual bool IsPtrValid() const {
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    return GetDevicePtr() != nullptr;
  }

  bool IsNotNeedAlloc() const { return IsPtrValid() || TEST_FLAG(flag(), device::kDeviceAddressFlagNotUsed); }

  using SyncUserDataHandler = void (*)(DeviceAddress *const device_address);
  // Return the valid device ptr.
  virtual void *GetValidPtr(size_t) {
    if (user_data() == nullptr || (!need_sync_user_data_)) {
      return GetDevicePtr();
    }
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    if (!need_sync_user_data_) {
      return GetDevicePtr();
    }
    auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
    if (sync_handler == nullptr) {
      MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
      return GetDevicePtr();
    }
    (*sync_handler)(this);
    need_sync_user_data_ = false;
    return GetDevicePtr();
  }

  inline void TouchSyncHandler() {
    if (!need_sync_user_data_ || kernel_tensor_->user_data() == nullptr) {
      return;
    }
    std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
    auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
    if (sync_handler == nullptr) {
      MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
      return;
    }
    (*sync_handler)(this);
    need_sync_user_data_ = false;
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

    other->set_from_mem_pool(this->from_mem_pool());
    other->set_deleter(deleter());
    other->set_need_sync_user_data(need_sync_user_data_);
    SetDevicePtr(nullptr);
    this->set_from_mem_pool(false);
    deleter_ = nullptr;
    kernel_tensor()->set_task_id_on_stream(other->kernel_tensor()->task_id_on_stream());
    kernel_tensor()->set_managed_by_somas(other->kernel_tensor()->managed_by_somas());
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

  // For output of pyexecute kernel, the input data is stored in user data and the handler is used to sync data from
  // user data to device ptr.
  bool need_sync_user_data() { return need_sync_user_data_; }
  void set_need_sync_user_data(bool need_sync_user_data) { need_sync_user_data_ = need_sync_user_data; }

  const PointerRefCountPtr &pointer_ref_count() const { return address_common_->pointer_ref_count_; }
  void set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt) {
    MS_EXCEPTION_IF_NULL(ptr_ref_cnt);
    address_common_->pointer_ref_count_ = ptr_ref_cnt;
  }

  void set_is_view(bool is_view) { is_view_ = is_view; }
  bool is_view() const { return is_view_; }
  AddressCommonPtr address_common() const { return address_common_; }

 protected:
  KernelTensorPtr kernel_tensor_{nullptr};
  // address basic info
  AddressCommonPtr address_common_{nullptr};
  size_t size() const { return address_common_->size_; }

  void *GetDevicePtr() const { return address_common_->pointer_ref_count_->ptr(); }
  void SetDevicePtr(void *ptr) const { address_common_->pointer_ref_count_->set_ptr(ptr); }

  void SetTypeId(TypeId type) const { address_common_->dtype_id_ = type; }

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

  bool from_persistent_mem_{false};
  bool need_recycle_{false};

  // The padding type corresponds to data format.
  std::string padding_type_;

  // The device address flag.
  size_t flag_{0};

  // Indicating whether the address is the input of view op.
  // If yes, the device address cannot be reused with the host address in CPU.
  bool is_view_{false};

  // The flag identify where data is stored
  mutable DeviceAddressStatus status_{DeviceAddressStatus::kInDevice};
  // Handler for sync data from user data.
  bool need_sync_user_data_{false};
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
