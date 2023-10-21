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
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "mindspore/core/ops/base_operator.h"
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"

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
// The memory info of kernel launch.
struct KernelLaunchInfo {
  AddressPtrList inputs_;
  AddressPtrList outputs_;
  AddressPtrList workspaces_;
};
struct TensorInfo {
  mindspore::Format format;
  abstract::AbstractTensorPtr base_;
  std::vector<int64_t> device_shape_adaptively;
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
// we extend KernelTensor to represent tensor/map_tensor/list/tuple/scalar data type.
class BACKEND_EXPORT KernelTensor {
 public:
  KernelTensor() = default;
  ~KernelTensor() = default;
  KernelTensor(const KernelTensor &copy_tensor) {
    meta_type_ = copy_tensor.meta_type_;
    meta_ = copy_tensor.meta_;
    data_ = copy_tensor.data_;
    host_data_ = copy_tensor.host_data_;
    device_id_ = copy_tensor.device_id_;
  }
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
  // If real type is not a single shape vector, it will return empty.
  ShapeVector GetShapeVector() const;
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
  void SetShapeVector(const ShapeVector &shape) const;

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
  // deprecated field for dynamic shape
  const ShapeVector &GetDeviceShapeAdaptively() const;
  void SetDeviceShapeAdaptively(const ShapeVector &device_shape_adaptively);
  int32_t GetDeviceId() const { return device_id_; }
  void SetDeviceId(int32_t device_id) { device_id_ = device_id; }

 private:
  TypeId meta_type_{kObjectTypeTensorType};
  // meta is a type-safe union of TensorInfo, ScalarInfo, TupleInfo, ListInfo.
  std::variant<TensorInfo, ScalarInfo, TupleInfo, ListInfo> meta_{TensorInfo()};
  AddressPtr data_{nullptr};                             // Device data address.
  AddressPtr host_data_{nullptr};                        // Host data address.
  std::unique_ptr<uint8_t[]> dyn_output_data_{nullptr};  // Create new output memory buffer for dynamic output
  string GetAbstractName() const;
  int32_t device_id_{0};
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
  KernelMod() {}
  explicit KernelMod(const BaseOperatorPtr &op) : op_(op) {}
  virtual ~KernelMod() = default;
  // Initialization for the kernel mod.
  inline bool Init_(const BaseOperatorPtr &op, const std::vector<KernelTensorPtr> &inputs,
                    const std::vector<KernelTensorPtr> &outputs) {
    MS_EXCEPTION_IF_NULL(op);
    this->op_ = op;
    inputs_ = inputs;
    outputs_ = outputs;
    return Init(op, inputs, outputs);
  }
  inline std::vector<KernelTensorPtr> &GetInputs() { return inputs_; }
  inline std::vector<KernelTensorPtr> &GetOutputs() { return outputs_; }
  virtual void SetInputSizeList(const std::vector<size_t> &size_list) { input_size_list_ = size_list; }
  virtual void SetOutputSizeList(const std::vector<size_t> &size_list) { output_size_list_ = size_list; }
  virtual void SetWorkspaceSizeList(const std::vector<size_t> &size_list) { workspace_size_list_ = size_list; }
  virtual const std::vector<size_t> &GetInputSizeList() const { return input_size_list_; }
  virtual const std::vector<size_t> &GetOutputSizeList() const { return output_size_list_; }
  virtual const std::vector<size_t> &GetWorkspaceSizeList() const { return workspace_size_list_; }
  virtual bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                      const std::vector<AddressPtr> &outputs, void *stream_ptr) = 0;
  virtual std::vector<size_t> GenParameters() { return {}; }
  virtual void GenAtomicInitInfo(AtomicInitInfo *info) {}
  virtual std::vector<KernelAttr> GetOpSupport() = 0;
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
  // Some kernels, e.g., Shape/Reshape, don't use some input addresses in the kernel launch.
  virtual std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const { return {}; }
  void set_unique_name(const std::string &unique_name) { unique_name_ = unique_name; }
  void set_fullname(const std::string &fullname) { fullname_ = fullname; }
  void set_is_monad(bool is_monad) { is_monad_ = is_monad; }
  void set_inputs_addr(const std::vector<AddressPtr> &addr) { inputs_addr_ = addr; }
  void set_workspaces_addr(const std::vector<AddressPtr> &addr) { workspaces_addr_ = addr; }
  void set_outputs_addr(const std::vector<AddressPtr> &addr) { outputs_addr_ = addr; }
  // User data is the extra dat-a required when the kernel is launched, It will be set before launch by runtime.
  virtual void set_input_user_data(UserData *user_data, size_t input_index) {}
  virtual void set_output_user_data(UserData *user_data, size_t output_index) {}
  // If output of kernel has a user_data, it needs to return true, and the framework will create user_data for it.
  virtual bool need_user_data() const { return false; }
  const std::vector<AddressPtr> &GetInputsAddr() const { return inputs_addr_; }
  const std::vector<AddressPtr> &GetWorkSpacesAddr() const { return workspaces_addr_; }
  const std::vector<AddressPtr> &GetOutputsAddr() const { return outputs_addr_; }
  void SetDevicedId(uint32_t device_id) { device_id_ = device_id; }
  virtual enum KernelModType GetKernelModType() const { return KernelModType::KernelMod; }
  bool Launch(const KernelLaunchInfo &kernel_launch_address, void *stream_ptr) {
    return Launch(kernel_launch_address.inputs_, kernel_launch_address.workspaces_, kernel_launch_address.outputs_,
                  stream_ptr);
  }
  int32_t task_id() const { return task_id_; }
  bool use_kernel_tensor() const { return use_kernel_tensor_; }
  void set_use_kernel_tensor(bool use_kernel_tensor) { use_kernel_tensor_ = use_kernel_tensor; }

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
  std::string kernel_name_;
  std::string unique_name_;
  std::string fullname_;
  bool is_monad_{false};
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  bool is_need_retrieve_output_shape_ = false;
  uint32_t device_id_ = 0;
  int32_t task_id_ = -1;
  std::vector<AddressPtr> inputs_addr_;
  std::vector<AddressPtr> workspaces_addr_;
  std::vector<AddressPtr> outputs_addr_;
  BaseOperatorPtr op_;
  std::vector<KernelTensorPtr> inputs_;
  std::vector<KernelTensorPtr> outputs_;
  bool use_kernel_tensor_{false};
};
using KernelModPtr = std::shared_ptr<KernelMod>;

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

BACKEND_EXPORT std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensorPtr> &tensors);
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
