/**
 * Copyright 2019-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_
#define MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_

#include <cstddef>
#include <atomic>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>
#include <algorithm>
#include <functional>
#include "abstract/abstract_value.h"
#include "mindapi/base/format.h"
#include "include/backend/visible.h"
#include "include/utils/utils.h"
#include "ir/anf.h"
#include "ir/tensor.h"
#include "ir/kernel_tensor_value.h"
#include "runtime/hardware_abstract/visible.h"
#include "device_address/device_address.h"

namespace mindspore {
namespace kernel {
using abstract::AbstractBase;
using DeviceAddress = device::DeviceAddress;
using DeviceAddressPtr = device::DeviceAddressPtr;

template <typename T>
struct ValidContainerChecker : std::false_type {};

// A ValidContainerChecker's specialization to detect whether the type is std::vector whose element is scalar.
template <typename... Args>
struct ValidContainerChecker<std::vector<Args...>> : std::true_type {};

// A ValidContainerChecker's specialization to detect whether the type is std::string.
template <>
struct ValidContainerChecker<std::string> : std::true_type {};

// A wrapper used to check the types std::string and std::vector.
template <typename T>
struct IsValidContainer {
  static constexpr bool value = ValidContainerChecker<std::decay_t<T>>::value;
};

// Used to encapsulate host-side related data structures in KernelTensor.
struct KernelHostInfo {
  KernelHostInfo() = default;

  KernelHostInfo(const KernelHostInfo &other);
  KernelHostInfo &operator=(const KernelHostInfo &other) = delete;

  // The shape vector transformed according `shape_vector_` and `format_` is generally used on the operator side.
  // Operators on different platforms may require different format and shape information.
  ShapeVector shape_vector_after_format_trasform_{};

  // Make shape transform related interfaces thread-safe.
  std::mutex shape_transform_mutex_;

  // The object enum type id of the KernelTensor.
  TypeId type_id_{kTypeUnknown};

  // Saves the contents after the value is converted to continuous memory storage.
  KernelTensorValuePtr kernel_tensor_value_{nullptr};

  // Make GetValue related interfaces thread-safe.
  std::mutex value_mutex_;
};

struct Address {
  Address() : addr(nullptr), size(0) {}
  Address(void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;

// RefCount is used to express reference between kernel tensors.
struct RefCount {
  RefCount() = default;
  ~RefCount() = default;

  std::string ToString() const {
    std::ostringstream ofs;
    ofs << this << " origin ref count:" << original_ref_count_ << " ref count:" << ref_count_
        << " dynamic ref count:" << dynamic_ref_count_ << " new ref count:" << new_ref_count_
        << " is ptr persisted:" << is_ptr_persisted_;
    return ofs.str();
  }

  // The static reference count, the value can be calculated at compile phase.
  size_t original_ref_count_{1};
  // The current reference count value, it will be decreased in the running, and reset by original_ref_count_ when it is
  // zero.
  std::atomic<size_t> ref_count_{1};

  std::atomic<size_t> new_ref_count_{0};

  // The dynamic reference count, the value can be calculated at compile phase.
  std::atomic_int32_t dynamic_ref_count_{INT32_MAX};
  // The device address of the node that owns the device address cannot be updated and replaced.
  // Application scenario: set to true when the hardware execution mode requires that ptr cannot be changed during
  // execution.
  bool is_ptr_persisted_{false};
};
using RefCountPtr = std::shared_ptr<RefCount>;

// KernelTensor is used to express input and output parameters of kernels.
// KernelTensor is a generalized Tensor semantics, which can represent not only Tensor, but also the meta-information
// of Scalar, Tuple, List and other data structures. It saves the shape, type, value and format information required by
// operators Infer and Launch, and provides related Get/Set interfaces.
class RUNTIME_HARDWARE_EXPORT KernelTensor : public AbstractBase {
 public:
  using Deleter = DevicePointer::Deleter;
  using ContinuousKernelTensorsPtr = std::shared_ptr<std::vector<std::weak_ptr<KernelTensor>>>;

  KernelTensor();
  ~KernelTensor() = default;

  // Constructor of KernelTensor by shape, type, value.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Constructor of KernelTensor by device info.
  KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info, TypeId dtype_id,
               const ShapeVector &host_shape, const UserDataPtr &user_data = nullptr);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info,
               const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value, void *device_ptr,
               size_t size, const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
               const string &device_name, const UserDataPtr &user_data = nullptr);

  // Constructor of KernelTensor by shape, type, value storage info and device info.
  KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info,
               const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  explicit KernelTensor(const DeviceAddressPtr &device_address, const TensorStorageInfoPtr &storage_info = nullptr)
      : KernelTensor(device_address, storage_info, nullptr, nullptr, nullptr) {}

  KernelTensor(const KernelTensor &other);
  KernelTensor &operator=(const KernelTensor &) = delete;

  MS_DECLARE_PARENT(KernelTensor, AbstractBase);

  std::string ToString() const;
  // Get the base shape for Tensor/Sequence/Scalar.
  abstract::BaseShapePtr GetShape() const override { return shape_; }

  // Set the base shape for Tensor/Sequence/Scalar.
  // Note: for performance, the function `SetShape` uses type_id_, so need to SetType first.
  void SetShape(const abstract::BaseShapePtr &shape);

  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const { return shape_vector_; }

  // Set the shape vector for Tensor/Sequence/Scalar.
  void SetShapeVector(const ShapeVector &shape_vector);

  // Set the shape vector for Tensor/Sequence/Scalar with rvalue.
  void SetShapeVector(ShapeVector &&shape_vector);

  // Get the device shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetDeviceShapeVector() const;

  // Get the object type of the KernelTensor.
  TypePtr GetType() const override { return type_; }

  // Set the type for the KernelTensor.
  void SetType(const TypePtr &type);

  // Check whether the host info exists.
  bool host_info_exist() const { return host_info_ != nullptr; }

  // Set host info after construct
  void SetHostInfo(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Get the object enum type id of the KernelTensor.
  TypeId type_id() const {
    MS_EXCEPTION_IF_NULL(host_info_);
    return host_info_->type_id_;
  }

  // Get the data enum type id of the KernelTensor.
  TypeId dtype_id() const { return dtype_id_; }

  // Set the data enum type id of the KernelTensor.
  void set_dtype_id(TypeId dtype_id) { dtype_id_ = dtype_id; }

  // Set the value for the KernelTensor.
  void SetValue(const ValuePtr &value) { value_ = value; }

  // Get the value of the KernelTensor.
  ValuePtr GetValue() const override;

  // Get the address of the value converted to continuous memory storage.
  const void *GetValuePtr();

  // Get the value in KernelTensor, return it if there is specific value, otherwise throw an exception.
  template <typename T>
  T GetValueWithCheck() {
    auto value_opt = GetValue<T>();
    if (!value_opt.has_value()) {
      MS_LOG(EXCEPTION)
        << "Get value failed, there is no any value in KernelTensor."
           "Here are the possible reasons:"
           "1. When the operator KernelMod is registered, the data type is not correct, such as Scalar or Tuple, "
           "but is registered as Tensor."
           "2. If the KernelMod is registered correctly, it may be an attempt to GetValue the output of the "
           "previous operator. During compilation, the output of the operator has no value. You can check the ir "
           "file to see if the input for the current operator value is from an operator.";
    }
    return value_opt.value();
  }

  // Get the scalar value store in KernelTensor if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    MS_EXCEPTION_IF_NULL(host_info_);
    std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid scalar value.";
      return std::nullopt;
    }

    if (!SetKernelTensorValue()) {
      return std::nullopt;
    }
    MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
    MS_EXCEPTION_IF_CHECK_FAIL((host_info_->kernel_tensor_value_->GetDataSize() == sizeof(T)),
                               "The data size in kernel tensor value which contains a scalar [" +
                                 std::to_string(host_info_->kernel_tensor_value_->GetDataSize()) +
                                 "] is not equal to the data type size [" + std::to_string(sizeof(T)) + "]");

    const T *data_ptr = reinterpret_cast<const T *>(host_info_->kernel_tensor_value_->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);
    return *data_ptr;
  }

  // Get the std::vector/std::string value store in KernelTensor if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<IsValidContainer<T>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    if (!std::is_scalar_v<typename T::value_type>) {
      MS_LOG(EXCEPTION) << "The element of std::vector to get kernel tensor's value should be scalar type.";
    }
    MS_EXCEPTION_IF_NULL(host_info_);
    std::lock_guard<std::mutex> lock(host_info_->value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid value for vector or string.";
      return std::nullopt;
    }

    if (!SetKernelTensorValue()) {
      return std::nullopt;
    }
    MS_EXCEPTION_IF_NULL(host_info_->kernel_tensor_value_);
    size_t element_num = host_info_->kernel_tensor_value_->GetDataSize() / sizeof(typename T::value_type);
    if (element_num == 0) {
      return T();
    }
    const typename T::value_type *data_ptr =
      reinterpret_cast<const typename T::value_type *>(host_info_->kernel_tensor_value_->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);

    return T(data_ptr, data_ptr + element_num);
  }

  // Get the value stored in KernelTensor for type which is not scalar, std::vector or std::string if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<!IsValidContainer<T>::value && !std::is_pointer_v<T> &&
                                                !std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    if (dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid value.";
      return std::nullopt;
    }
    if (value_ && !value_->isa<ValueAny>()) {
      return mindspore::GetValue<T>(value_);
    }
    return std::nullopt;
  }

  // Get the value in KernelTensor, return it if there is specific value, otherwise throw an exception.
  template <typename T>
  std::optional<T> GetOptionalValueWithCheck() {
    if (value_ && value_->isa<None>()) {
      return std::nullopt;
    }
    return GetValueWithCheck<T>();
  }

  // Get the data format.
  mindspore::Format format() const;

  // Set the data format.
  void set_format(mindspore::Format format);

  // Get the data format of string type.
  std::string GetStringFormat() const;

  // Set the data format of string type.
  void SetStringFormat(const std::string &format);

  //  Set the pointer and reference count to nullptr, resource reclaiming of the device pointer is automatically
  //  released.
  void ReleaseDeviceRes() { device_address_->set_device_pointer(nullptr); }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { device_address_->device_pointer()->set_deleter(deleter); }

  void set_allocator(std::shared_ptr<AddressAllocator> allocator) {
    device_address_->device_pointer()->set_allocator(allocator);
  }

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const { return device_address_->device_pointer()->ptr(); }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr);

  // Get the memory size in byte of the KernelTensor.
  size_t size() const { return device_address_->size(); }

  // Set the memory size in byte of the KernelTensor.
  void set_size(size_t size) { device_address_->SetSize(size); }

  // Get device target name, such "GPU","Ascend".
  device::DeviceType GetDeviceType() const { return device_address_->GetDeviceType(); }

  // Set device target name, such "GPU","Ascend".
  void SetDeviceType(const device::DeviceType &device_type) { device_address_->SetDeviceType(device_type); }

  // Get device id.
  uint32_t device_id() const { return device_address_->device_id(); }

  // Get logical stream id.
  uint32_t stream_id() const { return device_address_->stream_id(); }

  // Set logical stream id.
  void set_stream_id(uint32_t stream_id) { device_address_->set_stream_id(stream_id); }

  // Get logical stream id for memory alloc.
  uint32_t alloc_stream_id() const { return alloc_stream_id_; }

  // Set logical stream id for memory alloc.
  void set_alloc_stream_id(uint32_t stream_id) { alloc_stream_id_ = stream_id; }

  // Get task id on stream.
  std::shared_ptr<int64_t> task_id_on_stream() const { return task_id_on_stream_; }

  // Set task id on stream.
  void set_task_id_on_stream(const std::shared_ptr<int64_t> &task_id_on_stream) {
    task_id_on_stream_ = task_id_on_stream;
  }

  bool managed_by_somas() const { return managed_by_somas_; }

  void set_managed_by_somas(bool managed_by_somas) { managed_by_somas_ = managed_by_somas; }

  ContinuousKernelTensorsPtr continuous_kernel_tensors() const;
  void set_continuous_kernel_tensors(const ContinuousKernelTensorsPtr &continuous_kernel_tensors);

  // Get user data maintained by the KernelTensor.
  UserDataPtr user_data() const { return user_data_; }
  // Set user data to the KernelTensor.
  void set_user_data(const UserDataPtr &user_data) { user_data_ = user_data; }

  // For output of pyexecute kernel, the input data is stored in user data and the handler is used to sync data from
  // user data to device ptr.
  bool need_sync_user_data() { return need_sync_user_data_; }
  void set_need_sync_user_data(bool need_sync_user_data) { need_sync_user_data_ = need_sync_user_data; }

  // Return the valid device ptr.
  void *GetValidPtr(size_t);

  using SyncUserDataHandler = void (*)(KernelTensor *const kernel_tensor);

  inline void TouchSyncHandler() {
    if (!need_sync_user_data_ || user_data() == nullptr) {
      return;
    }
    std::lock_guard<std::mutex> lock(ptr_mutex_);
    auto sync_handler = user_data()->get<SyncUserDataHandler>(kSyncUserDataHandler);
    if (sync_handler == nullptr) {
      MS_LOG(WARNING) << "For device address:" << this << ", the sync user data handler is null.";
      return;
    }
    (*sync_handler)(this);
    need_sync_user_data_ = false;
  }

  bool is_ptr_persisted() const;
  void set_is_ptr_persisted(bool is_ptr_persisted);
  // Clone a new KernelTensor from this.
  std::shared_ptr<KernelTensor> CloneKernelTensor() { return std::make_shared<KernelTensor>(*this); }

  // Check whether the shape is dynamic shape(contains dim which is less than 0).
  bool IsDynamicShape() const;

  // Check whether the KernelTensor is from a constant variable(such as ValueNode).
  inline bool IsConstValue() const { return (value_ != nullptr) && !(value_->isa<ValueAny>()); }

  // The following four methods are only used in the Lite framework.
  // Get the device data address(pointer and size).
  AddressPtr GetData() const { return data_; }
  // Set the device data address(pointer and size).
  void SetData(const AddressPtr &data) { data_ = data; }
  // Get the host data address(pointer and size).
  AddressPtr GetHostData() const { return host_data_; }
  // Set the host data address(pointer and size).
  void SetHostData(const AddressPtr &data) { host_data_ = data; }

  // max shape is only used in compute-depended ops
  ShapeVector GetMaxShape() const;

  const TensorStorageInfoPtr tensor_storage_info() const { return tensor_storage_info_; }
  void set_tensor_storage_info(const TensorStorageInfoPtr &storage_info) { tensor_storage_info_ = storage_info; }

  size_t GetSize() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->GetSize();
  }

  // The interface of flag.
  size_t flag() const;
  void set_flag(size_t flag);
  void UpdateFlag(size_t flag);
  void ClearFlag(size_t flag);
  bool IsNotNeedAlloc() const;
  bool IsNotNeedAllocWOLock() const;

  const DeviceAddressPtr &device_address() const;
  void set_device_address(const DeviceAddressPtr &device_address);

  void Swap(KernelTensor *other) {
    MS_EXCEPTION_IF_NULL(other);
    auto other_device_address = other->device_address().get();
    MS_EXCEPTION_IF_NULL(device_address_);
    device_address_->Swap(other_device_address);
    other->set_task_id_on_stream(task_id_on_stream());
    other->set_need_sync_user_data(need_sync_user_data());
  }

  // Return whether KernelTensor has a valid ptr.
  bool IsPtrValid() const {
    MS_EXCEPTION_IF_NULL(device_address_);
    return device_address_->IsPtrValid();
  }

  // Get pointer and reference count.
  const DevicePointerPtr &device_pointer() const { return device_address_->device_pointer(); }

  // Set pointer and reference count.
  void set_pointer_ref_count(KernelTensor *const other);

  void IncreaseNewRefCount(const std::string &op_name, size_t i = 1);
  size_t DecreaseNewRefCount(const std::string &op_name);

  // The related interface of static reference count operation.
  void set_original_ref_count(size_t original_ref_count);
  size_t original_ref_count() const;
  void set_ref_count(size_t ref_count);
  size_t ref_count() const;
  void IncreaseOriginalRefCount();
  void DecreaseOriginalRefCount();

  void IncreaseRefCount(size_t increase_cnt);
  size_t DecreaseRefCount();
  void ResetRefCount();

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_count);
  int32_t dynamic_ref_count() const;

  void IncreaseDynamicRefCount(const std::string &op_object, int32_t increase_cnt);
  void IncreaseDynamicRefCount(const std::string &op_object);
  int32_t DecreaseDynamicRefCount(const std::string &op_object);

  // New ref count interface.
  void IncreaseNewRefCount(size_t i = 1);
  size_t DecreaseNewRefCount();
  void set_new_ref_count(size_t new_ref_count);
  size_t new_ref_count() const;

 private:
  // This is a deprecated function in base class.
  BaseShapePtr BuildShape() const override {
    MS_LOG(EXCEPTION) << "Call deprecated function: BuildShape, Please use GetShape instead of BuildShape in "
                         "operators' infer functions in the `core/ops` directory.";
  }

  // This is a deprecated function in base class
  TypePtr BuildType() const override {
    MS_LOG(EXCEPTION) << "Call deprecated function: BuildType, Please use GetType instead of BuildType in "
                         "operators' infer functions in the `core/ops` directory.";
  }

  // Set the element data type to KernelTensor for Sequence type(Tuple or List).
  void SetSequenceDType(const TypePtr &element_type);

  // Synchronize value data from device to host side.
  bool SyncDataFromDeviceToHost() const;

  // Update the kernel_tensor_value from host or device data.
  bool SetKernelTensorValue() const;

  // Calculate memory size need by the KernelTensor.
  void CalculateMemSize();

  // Check whether need to transpose host infer shape to device shape.
  bool NeedTransposeToDeviceShape() const noexcept;

  // Transpose host infer shape to device shape according format.
  const ShapeVector &TransposeToDeviceShape() const;

  // If host info is not initialized in the constructor, initialize it when you need it, making sure that host info is
  // not empty when used.
  void CheckHostInfoValid();

  // The host-side related data in KernelTensor.
  // Note: To improve the performance of constructing KernelTensor, allow some constructors not to initialize host
  // info. If host info is not initialized in the constructor, it can be initialized when it is needed.
  std::unique_ptr<KernelHostInfo> host_info_{nullptr};

  // The launch index on stream managed by framework.
  std::shared_ptr<int64_t> task_id_on_stream_{nullptr};

  // The following two variables are only used in the Lite framework.
  // Device data address.
  AddressPtr data_{nullptr};
  // Host data address.
  AddressPtr host_data_{nullptr};

  // device address info
  DeviceAddressPtr device_address_{nullptr};
  ContinuousKernelTensorsPtr continuous_kernel_tensors_{nullptr};
  // The kernel tensor flag.
  size_t flag_{0};

  // The data enum type id of the KernelTensor.
  TypeId dtype_id_{kTypeUnknown};
  // The origin flatten shape vector for Tensor/Scalar/Tuple/List.
  // 1. For Tensor type, means its shape. For example, a Tensor with shape (8, 16), shape_vector_ is {8, 16}.
  // 2. For Scalar type, shape_vector_ is an empty ShapeVector, i.e. {}.
  // 3. For Tuple/List (all elements must be Tensor with same shape or Scalar) type, the shape_vector_
  // consists of the element number and the shape of element in Tuple/List. For example, if a Tuple of the structure
  // ((8,16), (8,16)) contains two Tensors of shape (8, 16), then shape_vector_ is {2, 8, 16}, 2 means elements
  // number in Tuple/List. A Tuple with a structure such as ((), ()) that contains two Scalar, the shape_vector_ of
  // this Tuple is {2}.
  ShapeVector shape_vector_{};
  Format format_{Format::DEFAULT_FORMAT};

  UserDataPtr user_data_{nullptr};
  // Handler for sync data from user data.
  bool need_sync_user_data_{false};
  // Thread lock for ptr_.
  mutable std::mutex ptr_mutex_;
  bool managed_by_somas_{false};
  RefCountPtr ref_cnt_;
  // stream id for memory alloc
  uint32_t alloc_stream_id_{0};

  // tensor storage info
  TensorStorageInfoPtr tensor_storage_info_{nullptr};
};

using KernelTensorPtr = std::shared_ptr<KernelTensor>;
using ContinuousKernelTensorsPtr = std::shared_ptr<std::vector<std::weak_ptr<KernelTensor>>>;

}  // namespace kernel

RUNTIME_HARDWARE_EXPORT bool SyncCopy(kernel::KernelTensor *const dst_kernel_tensor,
                                      kernel::KernelTensor *const src_kernel_tensor, size_t stream_id);
RUNTIME_HARDWARE_EXPORT bool SyncCopy(const tensor::TensorPtr &dst_tensor,
                                      kernel::KernelTensor *const src_kernel_tensor, size_t stream_id);
RUNTIME_HARDWARE_EXPORT bool SyncCopy(kernel::KernelTensor *const dst_kernel_tensor, tensor::Tensor *const src_tensor,
                                      size_t stream_id);

RUNTIME_HARDWARE_EXPORT bool AsyncCopy(kernel::KernelTensor *const dst_kernel_tensor,
                                       kernel::KernelTensor *const src_kernel_tensor, size_t stream_id,
                                       bool keep_src = true);
RUNTIME_HARDWARE_EXPORT bool AsyncCopy(kernel::KernelTensor *const dst_kernel_tensor, tensor::Tensor *const src_tensor,
                                       size_t stream_id, bool keep_src = true);
RUNTIME_HARDWARE_EXPORT bool AsyncCopy(const tensor::TensorPtr &dst_tensor,
                                       kernel::KernelTensor *const src_kernel_tensor, size_t stream_id,
                                       bool keep_src = true);
}  // namespace mindspore

#endif  // MINDSPORE_OPS_KERNEL_KERNEL_TENSOR_H_
