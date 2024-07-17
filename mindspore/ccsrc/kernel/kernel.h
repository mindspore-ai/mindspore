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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
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
#include <unordered_map>
#include "abstract/dshape.h"
#include "abstract/ops/primitive_infer_map.h"
#include "include/api/format.h"
#include "include/backend/visible.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/convert_utils.h"
#include "include/backend/device_synchronizer.h"
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/kernel_tensor_value.h"
#include "mindspore/core/ops/base_operator.h"
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "ops/op_name.h"
#include "kernel/format_utils.h"
#include "mindapi/base/types.h"

#ifdef _MSC_VER
#undef OPAQUE
#endif

#ifdef OPAQUE
#undef OPAQUE
#endif

namespace mindspore {
enum KernelType : int {
  UNKNOWN_KERNEL_TYPE = 0,
  AKG_KERNEL,
  AICPU_KERNEL,
  RT_KERNEL,
  HCCL_KERNEL,
  TBE_KERNEL,
  HOST_KERNEL,
  CPU_KERNEL,
  GPU_KERNEL,
  BISHENG_KERNEL,
  ACL_KERNEL,
  OPAPI_KERNEL,
  INTERNAL_KERNEL,
};

// PointerRefCount encapsulates pointer and reference count-related operations, and supports custom deleter to free
// resources. In Ref scenarios, KernelTensor of different DeviceAddress may hold the same PointerRefCount object.
class PointerRefCount {
 public:
  // The arguments are pointer and a bool variable that identifies whether pointer is from the memory pool.
  using Deleter = std::function<void(void *, bool)>;

  PointerRefCount() = default;
  explicit PointerRefCount(void *ptr) : ptr_(ptr) {}
  PointerRefCount(void *ptr, const Deleter &deleter) : ptr_(ptr), deleter_(deleter) {}

  PointerRefCount(const PointerRefCount &other)
      : ptr_(other.ptr_),
        original_ref_count_(other.original_ref_count_),
        ref_count_(other.ref_count_.load()),
        dynamic_ref_count_(other.dynamic_ref_count_.load()),
        deleter_(other.deleter_) {}

  ~PointerRefCount() {
    try {
      if (ptr_ != nullptr && deleter_) {
        deleter_(ptr_, from_mem_pool_);
      }
      ptr_ = nullptr;
    } catch (const std::exception &e) {
      MS_LOG(ERROR) << "PointerRefCount destructed failed: " << e.what();
    } catch (...) {
      MS_LOG(ERROR) << "PointerRefCount destructed failed.";
    }
  }

  // Get raw pointer.
  void *ptr() const { return ptr_; }
  // Set raw pointer.
  void set_ptr(void *ptr) { ptr_ = ptr; }

  // Get whether pointer in PointerRefCount is allocated from the memory pool.
  bool from_mem_pool() const { return from_mem_pool_; }
  // Set whether pointer in PointerRefCount is allocated from the memory pool.
  void set_from_mem_pool(bool from_mem_pool) { from_mem_pool_ = from_mem_pool; }

  // Increase ref count or dynamic ref count.
  size_t IncreaseCounter() {
    if (ref_count_ != SIZE_MAX) {
      return ++ref_count_;
    } else if (dynamic_ref_count_ != INT32_MAX) {
      return ++dynamic_ref_count_;
    }
    return SIZE_MAX;
  }
  // Decrease ref count or dynamic ref count.
  size_t DecreaseCounter() {
    if (ref_count_ != SIZE_MAX) {
      return --ref_count_;
    } else if (dynamic_ref_count_ != INT32_MAX) {
      return --dynamic_ref_count_;
    }
    return SIZE_MAX;
  }

  // The related interface of static reference count operation.
  void set_original_ref_count(size_t original_ref_count) { original_ref_count_ = original_ref_count; }
  size_t original_ref_count() const { return original_ref_count_; }
  void set_ref_count(size_t ref_count) { ref_count_ = ref_count; }
  size_t ref_count() const { return ref_count_.load(); }
  void IncreaseOriginalRefCount() {
    if (original_ref_count_ < SIZE_MAX) {
      original_ref_count_++;
    }
  }
  void DecreaseOriginalRefCount() {
    if ((original_ref_count_ < SIZE_MAX) && (original_ref_count_ > 0)) {
      original_ref_count_--;
    }
  }
  size_t DecreaseRefCount() { return --ref_count_; }
  void ResetRefCount() { ref_count_ = original_ref_count_; }

  // The related interface of dynamic reference count operation.
  void set_dynamic_ref_count(int32_t dynamic_ref_count) { dynamic_ref_count_ = dynamic_ref_count; }
  int32_t dynamic_ref_count() const { return dynamic_ref_count_; }
  void IncreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ < INT32_MAX) {
      (void)++dynamic_ref_count_;
      MS_LOG(DEBUG) << op_object << " increases dynamic ref count to:" << dynamic_ref_count_ << " for ptr:" << ptr();
    }
  }
  int32_t DecreaseDynamicRefCount(const std::string &op_object) {
    if (dynamic_ref_count_ <= 0) {
      MS_LOG(EXCEPTION) << "The dynamic reference count is invalid value:" << dynamic_ref_count_;
    }
    MS_LOG(DEBUG) << op_object << " The dynamic ref count decreases to:" << dynamic_ref_count_ << " for ptr:" << ptr();
    return --dynamic_ref_count_;
  }

  // Get pointer resource destructor.
  Deleter deleter() const { return deleter_; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { deleter_ = deleter; }

 private:
  void *ptr_{nullptr};

  // Whether ptr_  is allocated from the memory pool.
  bool from_mem_pool_{false};

  // The static reference count, the value can be calculated at compile phase.
  size_t original_ref_count_{1};
  // The current reference count value, it will be decreased in the running, and reset by original_ref_count_ when it is
  // zero.
  std::atomic<size_t> ref_count_{1};

  // The dynamic reference count, the value can be calculated at compile phase.
  std::atomic_int32_t dynamic_ref_count_{INT32_MAX};

  // The pointer resource destructor.
  Deleter deleter_;
};
using PointerRefCountPtr = std::shared_ptr<PointerRefCount>;

namespace kernel {

// Backend processor
enum Processor {
  UNKNOWN = -1,
  AICORE = 0,
  AICPU,
  CUDA,
  CPU,
  BISHENG,
};

struct AtomicInitInfo {
  std::vector<std::string> dtype_list;
  std::vector<int64_t> init_value_int64_list;
  std::vector<float> init_value_float_list;
};

/**
 * @brief base class for autotensor kernel and cce kernel.
 */
struct Address {
  Address() : addr(nullptr), size(0) {}
  Address(void *address_addr, size_t address_size) : addr(address_addr), size(address_size) {}
  void *addr;
  size_t size;
};
using AddressPtr = std::shared_ptr<Address>;
using AddressPtrList = std::vector<AddressPtr>;
using StreamType = void *;
using abstract::AbstractBase;
using device::DeviceSynchronizerPtr;
// The memory info of kernel launch.
struct KernelLaunchAddr {
  AddressPtrList inputs_;
  AddressPtrList outputs_;
  AddressPtrList workspaces_;
};
struct TensorInfo {
  mindspore::Format format;
  abstract::AbstractTensorPtr base_;
};
struct ScalarInfo {
  abstract::AbstractScalarPtr base_;
};
struct ListInfo {
  abstract::AbstractListPtr base_;
};
struct TupleInfo {
  abstract::AbstractTuplePtr base_;
};
using TensorInfoPtr = std::shared_ptr<TensorInfo>;
using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;

class KernelAttr;

// Used to encapsulate host-side related data structures in KernelTensor.
struct KernelHostInfo {
  KernelHostInfo() = default;

  KernelHostInfo(const KernelHostInfo &other);

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

// A template class used to detect whether it is a valid container.
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

struct AddressCommon {
  AddressCommon() { pointer_ref_count_ = std::make_shared<PointerRefCount>(); }
  AddressCommon(void *device_ptr, size_t size)
      : pointer_ref_count_(std::make_shared<PointerRefCount>(device_ptr)), size_(size) {}
  AddressCommon(void *device_ptr, size_t size, const ShapeVector &shape_vector, const Format &format, TypeId dtype_id,
                const std::string &device_name, uint32_t device_id, uint32_t stream_id = 0)
      : pointer_ref_count_(std::make_shared<PointerRefCount>(device_ptr)),
        stream_id_(stream_id),
        size_(size),
        format_(format),
        dtype_id_(dtype_id),
        device_name_(device_name),
        device_id_(device_id),
        shape_vector_(shape_vector) {}
  AddressCommon(const AddressCommon &other) {
    pointer_ref_count_ =
      other.pointer_ref_count_ != nullptr
        ? std::make_shared<PointerRefCount>(other.pointer_ref_count_->ptr(), other.pointer_ref_count_->deleter())
        : std::make_shared<PointerRefCount>();
    tensor_storage_info_ = other.tensor_storage_info_;
    stream_id_ = other.stream_id_;
    size_ = other.size_;
    format_ = other.format_;
    dtype_id_ = other.dtype_id_;
    device_id_ = other.device_id_;
    device_name_ = other.device_name_;
    dtype_id_ = other.dtype_id_;
    shape_vector_ = other.shape_vector_;
    managed_by_somas_ = other.managed_by_somas_;
  }
  PointerRefCountPtr pointer_ref_count_;
  TensorStorageInfoPtr tensor_storage_info_{nullptr};
  uint32_t stream_id_{0};
  size_t size_{0};
  Format format_{Format::DEFAULT_FORMAT};
  // The data enum type id of the KernelTensor.
  TypeId dtype_id_{kTypeUnknown};
  // The device target name, such as "GPU","Ascend".
  std::string device_name_;
  // Represents the device card id associated with the KernelTensor.
  uint32_t device_id_{0};
  // The origin flatten shape vector for Tensor/Scalar/Tuple/List.
  // 1. For Tensor type, means its shape. For example, a Tensor with shape (8, 16), shape_vector_ is {8, 16}.
  // 2. For Scalar type, shape_vector_ is an empty ShapeVector, i.e. {}.
  // 3. For Tuple/List (all elements must be Tensor with same shape or Scalar) type, the shape_vector_
  // consists of the element number and the shape of element in Tuple/List. For example, if a Tuple of the structure
  // ((8,16), (8,16)) contains two Tensors of shape (8, 16), then shape_vector_ is {2, 8, 16}, 2 means elements
  // number in Tuple/List. A Tuple with a structure such as ((), ()) that contains two Scalar, the shape_vector_ of
  // this Tuple is {2}.
  ShapeVector shape_vector_{};
  bool managed_by_somas_{false};
};
using AddressCommonPtr = std::shared_ptr<AddressCommon>;

// KernelTensor is used to express input and output parameters of kernels.
// KernelTensor is a generalized Tensor semantics, which can represent not only Tensor, but also the meta-information
// of Scalar, Tuple, List and other data structures. It saves the shape, type, value and format information required by
// operators Infer and Launch, and provides related Get/Set interfaces.
class BACKEND_EXPORT KernelTensor : public AbstractBase {
 public:
  using Deleter = PointerRefCount::Deleter;

  KernelTensor();
  ~KernelTensor() = default;
  explicit KernelTensor(const AddressCommonPtr &address_common) : address_common_(address_common) {}

  // Constructor of KernelTensor by shape, type, value.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Constructor of KernelTensor by device info.
  KernelTensor(void *device_ptr, size_t size, Format format, TypeId dtype_id, const ShapeVector &host_shape,
               const string &device_name, uint32_t device_id, const UserDataPtr &user_data = nullptr);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value, void *device_ptr,
               size_t size, const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
               const string &device_name, uint32_t device_id, const UserDataPtr &user_data = nullptr);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const AddressCommonPtr &address_common, const abstract::BaseShapePtr &shape, const TypePtr &type,
               const ValuePtr &value, const ShapeVector &host_shape, const UserDataPtr &user_data = nullptr);

  KernelTensor(const KernelTensor &other);

  MS_DECLARE_PARENT(KernelTensor, AbstractBase);

  // Get the base shape for Tensor/Sequence/Scalar.
  abstract::BaseShapePtr GetShape() const override { return shape_; }

  // Set the base shape for Tensor/Sequence/Scalar.
  // Note: for performance, the function `SetShape` uses type_id_, so need to SetType first.
  void SetShape(const abstract::BaseShapePtr &shape);

  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const { return address_common_->shape_vector_; }

  // Set the shape vector for Tensor/Sequence/Scalar.
  void SetShapeVector(const ShapeVector &shape_vector);

  // Set the shape vector for Tensor/Sequence/Scalar with rvalue.
  void SetShapeVector(ShapeVector &&shape_vector);

  // Get the device shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetDeviceShapeVector() const;

  // Get host shape for KernelTensor.
  const ShapeVector &host_shape() const { return host_shape_; }

  // Set host shape for KernelTensor.
  void set_host_shape(const ShapeVector &host_shape) { host_shape_ = host_shape; }

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
  TypeId dtype_id() const { return address_common_->dtype_id_; }

  // Set the data enum type id of the KernelTensor.
  void set_dtype_id(TypeId dtype_id) { address_common_->dtype_id_ = dtype_id; }

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
    if (address_common_->dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid scalar value.";
      return std::nullopt;
    } else if (value_ && !value_->isa<ValueAny>()) {
      if (host_info_->kernel_tensor_value_ == nullptr) {
        host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      }
    } else {
      // Sync value data from device.
      if (!SyncDataFromDeviceToHost()) {
        MS_LOG(ERROR) << "Sync data from device to host side failed";
        return std::nullopt;
      }
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
    if (address_common_->dtype_id_ == kMetaTypeNone) {
      MS_LOG(DEBUG) << "None type has no valid value for vector or string.";
      return std::nullopt;
    } else if (value_ && !value_->isa<ValueAny>()) {
      if (host_info_->kernel_tensor_value_ == nullptr) {
        host_info_->kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      }
    } else {
      // Sync value data from device.
      if (!SyncDataFromDeviceToHost()) {
        MS_LOG(ERROR) << "Sync data from device to host side failed";
        return std::nullopt;
      }
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
    if (address_common_->dtype_id_ == kMetaTypeNone) {
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
  mindspore::Format format() const { return address_common_->format_; }

  // Set the data format.
  void set_format(mindspore::Format format) { address_common_->format_ = format; }

  // Get the data format of string type.
  std::string GetStringFormat() const;

  // Set the data format of string type.
  void SetStringFormat(const std::string &format);

  // Get pointer and reference count.
  const PointerRefCountPtr &pointer_ref_count() const { return address_common_->pointer_ref_count_; }

  // Set pointer and reference count.
  void set_pointer_ref_count(const PointerRefCountPtr &ptr_ref_cnt) {
    address_common_->pointer_ref_count_ = ptr_ref_cnt;
  }

  //  Set the pointer and reference count to nullptr, resource reclaiming of the device pointer is automatically
  //  released.
  void ReleaseDeviceRes() { address_common_->pointer_ref_count_ = nullptr; }

  // Set pointer resource destructor.
  void set_deleter(const Deleter &deleter) { address_common_->pointer_ref_count_->set_deleter(deleter); }

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const { return address_common_->pointer_ref_count_->ptr(); }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr) { address_common_->pointer_ref_count_->set_ptr(ptr); }

  // Get the memory size in byte of the KernelTensor.
  size_t size() const { return address_common_->size_; }

  // Set the memory size in byte of the KernelTensor.
  void set_size(size_t size) { address_common_->size_ = size; }

  // Get device target name, such "GPU","Ascend".
  const std::string &device_name() const { return address_common_->device_name_; }

  // Set device target name, such "GPU","Ascend".
  void set_device_name(const std::string &device_name) { address_common_->device_name_ = device_name; }

  // Get device id.
  uint32_t device_id() const { return address_common_->device_id_; }

  // Set device id.
  void set_device_id(uint32_t device_id) { address_common_->device_id_ = device_id; }

  // Get logical stream id.
  uint32_t stream_id() const { return address_common_->stream_id_; }

  // Set logical stream id.
  void set_stream_id(uint32_t stream_id) { address_common_->stream_id_ = stream_id; }

  // Get task id on stream.
  std::shared_ptr<int64_t> task_id_on_stream() const { return task_id_on_stream_; }

  // Set task id on stream.
  void set_task_id_on_stream(const std::shared_ptr<int64_t> &task_id_on_stream) {
    task_id_on_stream_ = task_id_on_stream;
  }

  bool managed_by_somas() const { return address_common_->managed_by_somas_; }

  void set_managed_by_somas(bool managed_by_somas) { address_common_->managed_by_somas_ = managed_by_somas; }

  // Get user data maintained by the KernelTensor.
  const UserDataPtr &user_data() const { return user_data_; }

  // Set user data to the KernelTensor.
  void set_user_data(const UserDataPtr &user_data) { user_data_ = user_data; }

  // Set device synchronizer to the KernelTensor.
  void set_device_synchronizer(const DeviceSynchronizerPtr &device_synchronizer) {
    device_synchronizer_ = device_synchronizer;
  }

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

  const TensorStorageInfoPtr tensor_storage_info() const { return address_common_->tensor_storage_info_; }
  void set_tensor_storage_info(const TensorStorageInfoPtr &storage_info) {
    address_common_->tensor_storage_info_ = storage_info;
  }

  const AddressCommonPtr address_common() const { return address_common_; }
  void set_address_common(const AddressCommonPtr &address_common) { address_common_ = address_common; }

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
  // Note: To improve the performance of constructing KernelTensor, allow some constructors not to initialize host info.
  // If host info is not initialized in the constructor, it can be initialized when it is needed.
  std::unique_ptr<KernelHostInfo> host_info_{nullptr};

  // The launch index on stream managed by framework.
  std::shared_ptr<int64_t> task_id_on_stream_{nullptr};

  // The flatten shape(maybe after padding) vector.
  // Note: the 'host_shape_' will be repalced by 'shape_vector_' in the future.
  ShapeVector host_shape_{};

  // User data is the extra data required by the kernel or framework.
  UserDataPtr user_data_{nullptr};

  // For synchronizing data between device and host.
  DeviceSynchronizerPtr device_synchronizer_{nullptr};

  // The following two variables are only used in the Lite framework.
  // Device data address.
  AddressPtr data_{nullptr};
  // Host data address.
  AddressPtr host_data_{nullptr};

  // address basic info
  AddressCommonPtr address_common_{nullptr};
};
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

enum class KernelModType {
  Invalid = 0,
  KernelMod,
  GpuKernelMod,
  NativeGpuKernelMod,
  CpuKernelMod,
  NativeCpuKernelMod,
  HostKernelMod,
  DynamicAkgCpuKernelMod,
};

// The info of kernel launch.
struct KernelLaunchInfo {
  std::vector<KernelTensor *> inputs_;
  std::vector<KernelTensor *> outputs_;
  std::vector<KernelTensor *> workspaces_;
};

enum KernelErrorCode : int { KRET_OK = 0, KRET_RESIZE_FAILED = 1, KRET_UNKNOWN_SHAPE = 2, KRET_UNKNOWN_OUT_SHAPE = 3 };

class BACKEND_EXPORT KernelMod {
 public:
  KernelMod() = default;
  virtual ~KernelMod() = default;

  virtual std::vector<KernelAttr> GetOpSupport() = 0;

  virtual bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
    MS_LOG(EXCEPTION) << "The KernelMod[" << kernel_name_ << "] doesn't implement virtual function 'Init'";
  }

  inline bool Init(const PrimitivePtr &primitive, const std::vector<KernelTensor *> &inputs,
                   const std::vector<KernelTensor *> &outputs) {
    primitive_ = primitive;
    MS_EXCEPTION_IF_NULL(primitive_);
    kernel_name_ = primitive_->name();

    return Init(inputs, outputs);
  }

  // Resize() is for validating input/output shape and calculating the workspace size, framework will invoke this
  // routine after infer shape.
  virtual int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  virtual bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    return true;
  }

  // Some kernels, e.g., Unique, can only get its output shape after its computing finished.
  virtual bool IsNeedUpdateOutputShapeAndSize() { return false; }
  virtual void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {}

  // Some kernels, e.g., Shape/Reshape, don't use some input addresses in the kernel launch.
  virtual std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const { return {}; }

  void SetDevicedId(uint32_t device_id) { device_id_ = device_id; }
  virtual enum KernelModType GetKernelModType() const { return KernelModType::KernelMod; }

  virtual void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  virtual void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  virtual void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  const std::vector<size_t> &GetInputSizeList() const { MS_LOG(EXCEPTION) << "Call deprecated interface."; }
  virtual const std::vector<size_t> &GetOutputSizeList() const { return output_size_list_; }
  virtual const std::vector<size_t> &GetWorkspaceSizeList() const { return workspace_size_list_; }

  const PrimitivePtr &primitive() const { return primitive_; }
  const std::string &kernel_name() const { return kernel_name_; }

  virtual std::vector<size_t> GenParameters() { return {}; }
  virtual void GenAtomicInitInfo(AtomicInitInfo *info) {}

  virtual void set_unique_name(const std::string &unique_name) {
    MS_LOG(EXCEPTION) << "Call the method which doesn't implement";
  }

  virtual void set_fullname(const std::string &fullname) {
    MS_LOG(EXCEPTION) << "Call the method which doesn't implement";
  }

  virtual void set_is_monad(bool is_monad) { MS_LOG(EXCEPTION) << "Call the method which doesn't implement"; }

  // If output of kernel has a user_data, it needs to return true, and the framework will create user_data for it.
  virtual bool need_user_data() const { return false; }

  int32_t task_id() const { return task_id_; }
  bool use_kernel_tensor() const { return use_kernel_tensor_; }
  void set_use_kernel_tensor(bool use_kernel_tensor) { use_kernel_tensor_ = use_kernel_tensor; }

  uint32_t record_stream_id() const { return record_stream_id_; }
  void set_record_stream_id(uint32_t record_stream_id) { record_stream_id_ = record_stream_id; }

  virtual bool Finalize() { return true; }

 protected:
  bool IsValidShape(const ShapeVector &shape) const {
    if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
      return false;
    }
    return true;
  }

 protected:
  std::string kernel_name_;
  PrimitivePtr primitive_;
  uint32_t device_id_ = 0;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  int32_t task_id_ = -1;
  bool use_kernel_tensor_{false};
  uint32_t record_stream_id_{0};
};
using KernelModPtr = std::shared_ptr<KernelMod>;

template <typename T>
inline T *GetDeviceAddress(const std::vector<KernelTensor *> &addr_list, size_t index) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return nullptr;
  }

  if (addr_list[index] == nullptr) {
    MS_LOG(ERROR) << "The device address is nullptr, address index: " << index << ", and the length of 'addr_list' is "
                  << addr_list.size();
    return nullptr;
  }

  if (addr_list[index]->device_ptr() == nullptr) {
    MS_LOG(WARNING) << "The memory of device address is nullptr, address index: " << index
                    << ", and the length of 'addr_list' is " << addr_list.size();
    return nullptr;
  }

  // When the input is an empty tuple, the input size will be 0.
  if (addr_list[index]->size() == 0) {
    MS_LOG(INFO) << "The size of device address is zero, address index: " << index
                 << ", and the length of 'addr_list' is " << addr_list.size();
  }
  return reinterpret_cast<T *>(addr_list[index]->device_ptr());
}

BACKEND_EXPORT std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors);

BACKEND_EXPORT void ConvertLaunchInfoToAddr(const KernelLaunchInfo &launch_info, KernelLaunchAddr *mem_info);

template <typename T>
inline bool CheckNullInput(const std::vector<T> &input_shape) {
  // If input_shape.size() == 0, it means a scalar input; If input_shape.size() != 0 and input_shape contains 0,
  // it means a null input. Just return a null output.
  if (input_shape.size() != 0) {
    if (std::any_of(input_shape.begin(), input_shape.end(), [](T i) { return i == 0; })) {
      return true;
    }
  }
  return false;
}
#define CHECK_NULL_INPUT(input_shape) mindspore::kernel::CheckNullInput(input_shape)

template <typename T>
inline bool CheckShapeNull(const std::vector<T> &shape, std::string kernel_name, std::string param_name) {
  if (CHECK_NULL_INPUT(shape)) {
    MS_LOG(WARNING) << "For '" << kernel_name << "', the shape of " << param_name << " cannot contain zero, but got "
                    << shape;
    return true;
  }
  return false;
}

#define CHECK_SHAPE_NULL(shape, kernel_name, param_name) \
  mindspore::kernel::CheckShapeNull(shape, kernel_name, param_name)
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_KERNEL_H_
