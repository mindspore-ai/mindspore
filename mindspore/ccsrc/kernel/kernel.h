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
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <variant>
#include <vector>
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
};
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
struct KernelLaunchInfo {
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

// KernelTensor is used to express input and output parameters of kernels.
// KernelTensor is a generalized Tensor semantics, which can represent not only Tensor, but also the meta-information
// of Scalar, Tuple, List and other data structures. It saves the shape, type, value and format information required by
// operators Infer and Launch, and provides related Get/Set interfaces.
class BACKEND_EXPORT KernelTensor : public AbstractBase {
 public:
  KernelTensor() = default;
  ~KernelTensor() = default;

  // Constructor of KernelTensor by shape, type, value.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value);

  // Constructor of KernelTensor by device info.
  KernelTensor(void *device_ptr, size_t size, const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
               const string &device_name, uint32_t device_id, const UserDataPtr &user_data = nullptr);

  // Constructor of KernelTensor by shape, type, value and device info.
  KernelTensor(const abstract::BaseShapePtr &shape, const TypePtr &type, const ValuePtr &value, void *device_ptr,
               size_t size, const std::string &format, TypeId dtype_id, const ShapeVector &host_shape,
               const string &device_name, uint32_t device_id, const UserDataPtr &user_data = nullptr);

  KernelTensor(const KernelTensor &other);

  // Move constructor.
  KernelTensor(KernelTensor &&other) {
    shape_ = other.shape_;
    shape_vector_ = std::move(other.shape_vector_);

    type_ = other.type_;
    type_id_ = other.type_id_;
    dtype_ = other.dtype_;
    dtype_id_ = other.dtype_id_;

    value_ = other.value_;

    format_ = other.format_;
    device_ptr_ = other.device_ptr_;
    size_ = other.size_;
    device_name_ = std::move(other.device_name_);
    device_id_ = other.device_id_;
  }

  // Move assignment operator.
  KernelTensor &operator=(KernelTensor &&other) {
    shape_ = other.shape_;
    shape_vector_ = std::move(other.shape_vector_);

    type_ = other.type_;
    type_id_ = other.type_id_;
    dtype_ = other.dtype_;
    dtype_id_ = other.dtype_id_;

    value_ = other.value_;

    format_ = other.format_;
    device_ptr_ = other.device_ptr_;
    size_ = other.size_;
    device_name_ = std::move(other.device_name_);
    device_id_ = other.device_id_;

    return *this;
  }

  MS_DECLARE_PARENT(KernelTensor, AbstractBase);

  // Get the base shape for Tensor/Sequence/Scalar.
  abstract::BaseShapePtr GetShape() const override { return shape_; }

  // Set the base shape for Tensor/Sequence/Scalar.
  void SetShape(const abstract::BaseShapePtr &shape);

  // Get the shape vector for Tensor/Sequence/Scalar.
  const ShapeVector &GetShapeVector() const { return shape_vector_; }

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

  // Get the object enum type id of the KernelTensor.
  TypeId type_id() const { return type_id_; }

  // Get the data type of the KernelTensor.
  TypePtr dtype() const { return dtype_; }

  // Get the data enum type id of the KernelTensor.
  TypeId dtype_id() const { return dtype_id_; }

  // Set the data enum type id of the KernelTensor.
  void set_dtype_id(TypeId dtype_id) { dtype_id_ = dtype_id; }

  // Set the value for the KernelTensor.
  void SetValue(const ValuePtr &value) { value_ = value; }

  // Get the address of the value converted to continuous memory storage.
  const void *GetValuePtr() const {
    std::lock_guard<std::mutex> lock(value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (value_ && !value_->isa<ValueAny>()) {
      if (kernel_tensor_value_ == nullptr) {
        kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      }
      MS_EXCEPTION_IF_NULL(kernel_tensor_value_);
      return kernel_tensor_value_->GetDataPtr();
    }

    // Sync value data from device.
    if (!SyncDataFromDeviceToHost()) {
      MS_LOG(EXCEPTION) << "Sync data form device to host side failed";
    }
    return kernel_tensor_value_->GetDataPtr();
  }

  // Get the value of the KernelTensor.
  ValuePtr GetValue() const override {
    std::lock_guard<std::mutex> lock(value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (value_ && !value_->isa<ValueAny>()) {
      if (kernel_tensor_value_ == nullptr) {
        kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
        return kernel_tensor_value_ ? kernel_tensor_value_ : value_;
      }
      return kernel_tensor_value_;
    }

    // Sync value data from device.
    if (!SyncDataFromDeviceToHost()) {
      MS_LOG(EXCEPTION) << "Sync data form device to host side failed";
    }
    return kernel_tensor_value_;
  }

  // Get the scalar value store in KernelTensor if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    std::lock_guard<std::mutex> lock(value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (value_ && !value_->isa<ValueAny>()) {
      if (kernel_tensor_value_ == nullptr) {
        kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      }
    } else {
      // Sync value data from device.
      if (!SyncDataFromDeviceToHost()) {
        MS_LOG(ERROR) << "Sync data form device to host side failed";
        return std::nullopt;
      }
    }

    MS_EXCEPTION_IF_NULL(kernel_tensor_value_);
    MS_EXCEPTION_IF_CHECK_FAIL((kernel_tensor_value_->GetDataSize() == sizeof(T)),
                               "The data size in kernel tensor value which contains a scalar [" +
                                 std::to_string(kernel_tensor_value_->GetDataSize()) +
                                 "] is not equal to the data type size [" + std::to_string(sizeof(T)) + "]");

    const T *data_ptr = reinterpret_cast<const T *>(kernel_tensor_value_->GetDataPtr());
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
    std::lock_guard<std::mutex> lock(value_mutex_);

    // There is a origin value in KernelTensor(maybe come from a ValueNode).
    if (value_ && !value_->isa<ValueAny>()) {
      if (kernel_tensor_value_ == nullptr) {
        kernel_tensor_value_ = ConvertValueToKernelTensorValue(value_);
      }
    } else {
      // Sync value data from device.
      if (!SyncDataFromDeviceToHost()) {
        MS_LOG(ERROR) << "Sync data form device to host side failed";
        return std::nullopt;
      }
    }

    MS_EXCEPTION_IF_NULL(kernel_tensor_value_);
    size_t element_num = kernel_tensor_value_->GetDataSize() / sizeof(typename T::value_type);
    if (element_num == 0) {
      return T();
    }
    const typename T::value_type *data_ptr =
      reinterpret_cast<const typename T::value_type *>(kernel_tensor_value_->GetDataPtr());
    MS_EXCEPTION_IF_NULL(data_ptr);

    return T(data_ptr, data_ptr + element_num);
  }

  // Get the value stored in KernelTensor for type which is not scalar, std::vector or std::string if exists.
  // Return the optional contain value if the KernelTensor has value, otherwise nullopt.
  template <typename T, typename std::enable_if<!IsValidContainer<T>::value && !std::is_pointer_v<T> &&
                                                !std::is_scalar<std::decay_t<T>>::value>::type * = nullptr>
  std::optional<T> GetValue() {
    if (value_ && !value_->isa<ValueAny>()) {
      return mindspore::GetValue<T>(value_);
    }
    return std::nullopt;
  }

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

  // Get the data format.
  mindspore::Format format() const { return format_; }

  // Set the data format.
  void set_format(mindspore::Format format) { format_ = format; }

  // Get the data format of string type.
  std::string GetStringFormat() const;

  // Set the data format of string type.
  void SetStringFormat(const std::string &format);

  // Get the padding type of format.
  const std::string &padding_type() const;

  // Set the padding type of format.
  void set_padding_type(const std::string &padding_type);

  // Get pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr() const { return device_ptr_; }

  // Set pointer to the device side that corresponds to KernelTensor, used in runtime.
  void set_device_ptr(void *ptr) { device_ptr_ = ptr; }

  // Get the memory size in byte of the KernelTensor.
  size_t size() const { return size_; }

  // Set the memory size in byte of the KernelTensor.
  void set_size(size_t size) { size_ = size; }

  // Get device target name, such "GPU","Ascend".
  const std::string &device_name() const { return device_name_; }

  // Set device target name, such "GPU","Ascend".
  void set_device_name(const std::string &device_name) { device_name_ = device_name; }

  // Get device id.
  uint32_t device_id() const { return device_id_; }

  // Set device id.
  void set_device_id(uint32_t device_id) { device_id_ = device_id; }

  // Set logical stream id.
  void set_stream_id(size_t stream_id) { stream_id_ = stream_id; }

  // Get user data maintained by the KernelTensor.
  const UserDataPtr &user_data() const { return user_data_; }

  // Set user data to the KernelTensor.
  void set_user_data(const UserDataPtr &user_data) { user_data_ = user_data; }

  // Set device synchronizer to the KernelTensor.
  void set_device_synchronizer(const DeviceSynchronizerPtr &device_synchronizer) {
    device_synchronizer_ = device_synchronizer;
  }

  std::shared_ptr<KernelTensor> CloneKernelTensor() { return std::make_shared<KernelTensor>(*this); }

  // The following member methods are required by the old KernelTensor.
  KernelTensor &operator=(const KernelTensor &copy_tensor) {
    if (&copy_tensor == this) {
      return *this;
    }
    meta_type_ = copy_tensor.meta_type_;
    meta_ = copy_tensor.meta_;
    data_ = copy_tensor.data_;
    host_data_ = copy_tensor.host_data_;
    device_id_ = copy_tensor.device_id_;
    dyn_output_data_ = nullptr;
    return *this;
  }

  bool IsDynamicShape() const;
  size_t GetSizeInBytes() const;
  AddressPtr GetData() const { return data_; }
  AddressPtr GetHostData() const { return host_data_; }
  TypeId GetDtype() const;
  mindspore::Format GetFormat() const {
    if (meta_type_ == kObjectTypeTensorType) {
      const TensorInfo &info = std::get<TensorInfo>(meta_);
      return info.format;
    }
    return Format::DEFAULT_FORMAT;
  }
  TypeId GetMetaType() const { return meta_type_; }
  std::variant<TensorInfo, ScalarInfo, TupleInfo, ListInfo> GetMeta() const { return meta_; }
  // If real type is not a list or tuple tensor, it will return kTypeUnknown.
  std::vector<TypeId> GetListOrTupleDtype() const;

  // If real type is not a list or tuple shape vector, it will return empty.
  std::vector<ShapeVector> GetListOrTupleShapeVector() const;
  void SetData(const AddressPtr &data) { data_ = data; }
  void SetHostData(const AddressPtr &data) { host_data_ = data; }
  void SetDtype(const TypePtr &dtype);
  void SetFormat(mindspore::Format format) {
    TensorInfo &info = std::get<TensorInfo>(meta_);
    info.format = format;
  }
  void SetMetaType(const TypeId meta_type) { meta_type_ = meta_type; }

  // max shape is only used in compute-depended ops
  ShapeVector GetMaxShape() const;

  abstract::BaseShapePtr GetBaseShape() const;
  // If the shape need to be List or Tuple, `SetBaseShape` should be called.
  void SetBaseShape(const abstract::BaseShapePtr &base_shape);
  void SetTensorInfo(const TensorInfo &info) {
    meta_type_ = kObjectTypeTensorType;
    meta_ = info;
  }
  void SetScalarInfo(const ScalarInfo &info) {
    meta_type_ = kObjectTypeNumber;
    meta_ = info;
  }
  void SetTupleInfo(const TupleInfo &info) {
    meta_type_ = kObjectTypeTuple;
    meta_ = info;
  }
  void SetListInfo(const ListInfo &info) {
    meta_type_ = kObjectTypeList;
    meta_ = info;
  }
  void SetDynOutput(std::unique_ptr<uint8_t[]> &&new_buffer) { dyn_output_data_ = std::move(new_buffer); }
  uint8_t *GetDynOutput() const { return dyn_output_data_.get(); }
  int32_t GetDeviceId() const { return device_id_; }
  void SetDeviceId(int32_t device_id) { device_id_ = device_id; }

 private:
  // Set the element data type to KernelTensor for Sequence type(Tuple or List).
  void SetSequenceDType(const TypePtr &element_type);

  // Synchronize value data from device to host side.
  bool SyncDataFromDeviceToHost() const;

  // Calculate memory size need by the KernelTensor.
  void CalculateMemSize();

  // Check whether need to transpose host infer shape to device shape.
  bool NeedTransposeToDeviceShape() const { return format_ != Format::DEFAULT_FORMAT && format_ != Format::NCHW; }

  // Transpose host infer shape to device shape according format.
  void TransposeToDeviceShape() const;

  // The flatten shape vector for Tensor/Scalar/Tuple/List.
  // 1. For Tensor type, means its shape. For example, a Tensor with shape (8, 16), shape_vector_ is {8, 16}.
  // 2. For Scalar type, shape_vector_ is an empty ShapeVector, i.e. {}.
  // 3. For Tuple/List (all elements must be Tensor with same shape or Scalar) type, the shape_vector_
  // consists of the element number and the shape of element in Tuple/List. For example, if a Tuple of the structure
  // ((8,16), (8,16)) contains two Tensors of shape (8, 16), then shape_vector_ is {2, 8, 16}, 2 means elements
  // number in Tuple/List. A Tuple with a structure such as ((), ()) that contains two Scalar, the shape_vector_ of
  // this Tuple is {2}.
  ShapeVector shape_vector_{};

  // The shape of the device corresponding to 'shape_vector_'. For example, if format is NHWC, the shape of the device
  // and host may be different.
  mutable ShapeVector device_shape_vector_{};

  // Make GetDeviceShapeVector thread-safe.
  mutable std::mutex device_shape_mutex_;

  // The flatten shape(maybe after padding) vector.
  // Note: the 'host_shape_' will be repalced by 'shape_vector_' in the future.
  ShapeVector host_shape_{};

  // The object enum type id of the KernelTensor.
  TypeId type_id_{kTypeUnknown};

  // The data type of the KernelTensor.
  TypePtr dtype_{kTypeNone};
  // The data enum type id of the KernelTensor.
  TypeId dtype_id_{kTypeUnknown};

  // The memory in bytes of each element in KernelTensor.
  size_t element_size_in_bytes_{0};

  // Saves the contents after the value is converted to continuous memory storage.
  mutable KernelTensorValuePtr kernel_tensor_value_{nullptr};

  // The data format of the KernelTensor.
  mindspore::Format format_{Format::DEFAULT_FORMAT};

  // The padding type corresponds to data format.
  std::string padding_type_;

  // The pointer to the device side that corresponds to KernelTensor, used in runtime.
  void *device_ptr_{nullptr};

  // The memory size in byte of the KernelTensor.
  size_t size_{0};

  // The device target name, such "GPU","Ascend".
  std::string device_name_;

  // Represents the device card id associated with the KernelTensor.
  uint32_t device_id_;

  // The stream index in all stream array managed by Framework, starting form 0.
  size_t stream_id_{0};

  // User data is the extra data required by the kernel or framework.
  UserDataPtr user_data_{nullptr};

  // For synchronizing data between device and host.
  DeviceSynchronizerPtr device_synchronizer_{nullptr};

  // Make GetValue related interfaces thread-safe.
  mutable std::mutex value_mutex_;

  // The following member variables are required by the old KernelTensor.
  TypeId meta_type_{kObjectTypeTensorType};
  // meta is a type-safe union of TensorInfo, ScalarInfo, TupleInfo, ListInfo.
  std::variant<TensorInfo, ScalarInfo, TupleInfo, ListInfo> meta_{TensorInfo()};
  AddressPtr data_{nullptr};                             // Device data address.
  AddressPtr host_data_{nullptr};                        // Host data address.
  std::unique_ptr<uint8_t[]> dyn_output_data_{nullptr};  // Create new output memory buffer for dynamic output
  string GetAbstractName() const;
};
using KernelTensorPtr = std::shared_ptr<KernelTensor>;

enum class KernelModType {
  Invalid = 0,
  KernelMod,
  GpuKernelMod,
  NativeGpuKernelMod,
  DeprecatedNativeGpuKernelMod,
  CpuKernelMod,
  NativeCpuKernelMod,
  DeprecatedNativeCpuKernelMod,
  HostKernelMod,
  DynamicAkgCpuKernelMod,
};

enum KernelErrorCode : int { KRET_OK = 0, KRET_RESIZE_FAILED = 1, KRET_UNKNOWN_SHAPE = 2, KRET_UNKNOWN_OUT_SHAPE = 3 };

class BACKEND_EXPORT KernelMod {
 public:
  // ===========================New interface==========================================================
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

  const std::vector<AddressPtr> &GetInputsAddr() const { return inputs_addr_; }
  const std::vector<AddressPtr> &GetWorkSpacesAddr() const { return workspaces_addr_; }
  const std::vector<AddressPtr> &GetOutputsAddr() const { return outputs_addr_; }
  void set_inputs_addr(const std::vector<AddressPtr> &addr) { inputs_addr_ = addr; }
  void set_workspaces_addr(const std::vector<AddressPtr> &addr) { workspaces_addr_ = addr; }
  void set_outputs_addr(const std::vector<AddressPtr> &addr) { outputs_addr_ = addr; }

  const PrimitivePtr &primitive() const { return primitive_; }
  const std::string &kernel_name() const { return kernel_name_; }

  // =======================Old interface, will deleted after all kernel modified used new interface=================
  explicit KernelMod(const BaseOperatorPtr &op) : op_(op) {}
  // Initialization for the kernel mod.
  inline bool Init_(const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs) {
    // MS_EXCEPTION_IF_NULL(op);
    this->op_ = op;
    inputs_ = inputs;
    outputs_ = outputs;
    return Init(op, inputs, outputs);
  }
  inline std::vector<KernelTensorPtr> &GetInputs() { return inputs_; }
  inline std::vector<KernelTensorPtr> &GetOutputs() { return outputs_; }

  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    return true;
  }
  virtual std::vector<size_t> GenParameters() { return {}; }
  virtual void GenAtomicInitInfo(AtomicInitInfo *info) {}
  // Resize() is for validating input/output shape and calculating the workspace size, framework will invoke this
  // routine after infer shape.
  virtual int Resize(
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>());
  virtual int Resize(
    const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>());
  // Some kernels, e.g., Unique, can only get its output shape after its computing finished.
  virtual bool IsNeedRetrieveOutputShape() { return is_need_retrieve_output_shape_; }
  std::vector<KernelTensorPtr> RetrieveOutputShape() {
    SyncOutputShape();
    return outputs_;
  }

  virtual void set_unique_name(const std::string &unique_name) {
    MS_LOG(EXCEPTION) << "Call the method which doesn't implement";
  }

  virtual void set_fullname(const std::string &fullname) {
    MS_LOG(EXCEPTION) << "Call the method which doesn't implement";
  }

  virtual void set_is_monad(bool is_monad) { MS_LOG(EXCEPTION) << "Call the method which doesn't implement"; }

  // User data is the extra dat-a required when the kernel is launched, It will be set before launch by runtime.
  virtual void set_input_user_data(UserData *user_data, size_t input_index) {}
  virtual void set_output_user_data(UserData *user_data, size_t output_index) {}
  // If output of kernel has a user_data, it needs to return true, and the framework will create user_data for it.
  virtual bool need_user_data() const { return false; }

  bool Launch(const KernelLaunchInfo &kernel_launch_address, void *stream_ptr) {
    return Launch(kernel_launch_address.inputs_, kernel_launch_address.workspaces_, kernel_launch_address.outputs_,
                  stream_ptr);
  }
  int32_t task_id() const { return task_id_; }
  bool use_kernel_tensor() const { return use_kernel_tensor_; }
  void set_use_kernel_tensor(bool use_kernel_tensor) { use_kernel_tensor_ = use_kernel_tensor; }
  virtual bool Finalize() { return true; }

 protected:
  virtual bool Init(const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs) {
    return true;
  }
  virtual int Resize(
    const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs, const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>());
  bool IsValidShape(const ShapeVector &shape) const {
    if (std::any_of(shape.begin(), shape.end(), [](int64_t dim) { return dim < 0; })) {
      return false;
    }
    return true;
  }
  // some kernels' output shape can only get from its computing result, this routine is for getting output shape and
  // setting into outputs_.
  virtual void SyncOutputShape() {}

 protected:
  // ===========================New member==========================================================
  std::string kernel_name_;
  PrimitivePtr primitive_;
  uint32_t device_id_ = 0;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::vector<AddressPtr> inputs_addr_;
  std::vector<AddressPtr> workspaces_addr_;
  std::vector<AddressPtr> outputs_addr_;

  // =======================Old member, will deleted after all kernel modified used new interface=================
  bool is_need_retrieve_output_shape_ = false;
  int32_t task_id_ = -1;
  BaseOperatorPtr op_;
  std::vector<KernelTensorPtr> inputs_;
  std::vector<KernelTensorPtr> outputs_;
  bool use_kernel_tensor_{false};
};
using KernelModPtr = std::shared_ptr<KernelMod>;

// Delete after KernelMod rectified.
template <typename T>
inline T *GetDeviceAddress(const std::vector<AddressPtr> &addr_list, size_t index) {
  if (index >= addr_list.size()) {
    MS_LOG(ERROR) << "Address index(" << index << ") out of range(" << addr_list.size() << ")";
    return nullptr;
  }

  if (addr_list[index] == nullptr) {
    MS_LOG(ERROR) << "The device address is nullptr, address index: " << index << ", and the length of 'addr_list' is "
                  << addr_list.size();
    return nullptr;
  }

  if (addr_list[index]->addr == nullptr) {
    MS_LOG(WARNING) << "The memory of device address is nullptr, address index: " << index
                    << ", and the length of 'addr_list' is " << addr_list.size();
    return nullptr;
  }

  // When the input is an empty tuple, the input size will be 0.
  if (addr_list[index]->size == 0) {
    MS_LOG(INFO) << "The size of device address is zero, address index: " << index
                 << ", and the length of 'addr_list' is " << addr_list.size();
  }
  return reinterpret_cast<T *>(addr_list[index]->addr);
}

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
    return nullptr;
  }
  return reinterpret_cast<T *>(addr_list[index]->device_ptr());
}

// ===========================Old interface===========================
BACKEND_EXPORT std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensorPtr> &tensors);
// ===========================Old interface===========================
BACKEND_EXPORT std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors);

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
