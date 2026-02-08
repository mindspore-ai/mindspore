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
#ifndef MINDSPORE_OPS_COMMON_KERNEL_H_
#define MINDSPORE_OPS_COMMON_KERNEL_H_
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
#include "ir/format_utils.h"
#include "mindapi/base/format.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "ops/base_operator.h"
#include "nlohmann/json.hpp"
#include "utils/log_adapter.h"
#include "primitive/op_name.h"
#include "include/runtime/hardware_abstract/kernel_base/kernel_tensor.h"
#include "runtime/hardware_abstract/visible.h"

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
  SYMMETRIC_MEMORY_KERNEL,
  GE_KERNEL,
  ATB_KERNEL,
  CUSTOM_KERNEL,
};

namespace kernel {
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

struct AtomicInitInfo {
  std::vector<std::string> dtype_list;
  std::vector<int64_t> init_value_int64_list;
  std::vector<float> init_value_float_list;
};

using StreamType = void *;

// The memory info of kernel launch.
struct KernelLaunchAddr {
  AddressPtrList inputs_;
  AddressPtrList outputs_;
  AddressPtrList workspaces_;
};
using BaseOperatorPtr = std::shared_ptr<ops::BaseOperator>;

class KernelAttr;

enum KernelErrorCode : int { KRET_OK = 0, KRET_RESIZE_FAILED = 1, KRET_UNKNOWN_SHAPE = 2, KRET_UNKNOWN_OUT_SHAPE = 3 };

class RUNTIME_HARDWARE_EXPORT KernelMod {
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
  // UpdateWeights() is for update lora weights.
  virtual bool UpdateWeights(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
    return true;
  }

  // Some kernels, e.g., Unique, can only get its output shape after its computing finished.
  virtual bool IsNeedUpdateOutputShapeAndSize() { return false; }
  virtual void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {}

  // Some kernels, e.g., Shape/Reshape, don't use some input addresses in the kernel launch.
  virtual std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const { return {}; }

  // Some kernels, e.g., Send/Print, output is use less.
  virtual std::vector<size_t> GetUseLessOutputIdx() const { return {}; }

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
  PrimitivePtr primitive_{nullptr};
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

RUNTIME_HARDWARE_EXPORT std::vector<std::vector<int64_t>> GetShapes(const std::vector<KernelTensor *> &tensors);

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

#endif  // MINDSPORE_OPS_COMMON_KERNEL_H_
