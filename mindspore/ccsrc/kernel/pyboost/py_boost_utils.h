/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/common/utils/tensor_future.h"
#include "runtime/pynative/op_executor.h"
#include "mindspore/core/ops/view/view_strides_calculator.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/pyboost_kernel_extra_func.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT PyBoostUtils {
 public:
  static DeviceContext *GetDeviceContext(const std::string &device_type);
  static DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device);
  static AbstractBasePtr InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs);
  static void DispatchRun(const std::shared_ptr<pynative::PyBoostDeviceTask> &task);

  // Data convert
  static tensor::TensorPtr ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type);
  static tensor::TensorPtr ContiguousTensor(const tensor::TensorPtr &input_tensor);
  static device::DeviceAddressPtr ContiguousByDeviceAddress(const device::DeviceAddressPtr &old_device_address,
                                                            const TensorStorageInfoPtr &old_storage_info);

  // Create device address
  static device::DeviceAddressPtrList CreateWorkSpaceDeviceAddress(const KernelModPtr &kernel_mod,
                                                                   const device::DeviceContext *device_context,
                                                                   const std::string &op_name);

  // Create output tensors
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const tensor::TensorPtr &input, const TensorStorageInfoPtr &storage_info,
                                 std::vector<tensor::TensorPtr> *outputs);

  // Create input device address without kernel tensor
  template <typename... Args>
  static void PrepareOpInputs(DeviceContext *device_context, const Args &... args) {
    size_t index = 0;
    auto add_index = [&index]() { return index++; };
    (runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, add_index(), args), ...);
  }

  template <typename... T>
  static void MallocOpInputs(DeviceContext *device_context, const T &... args) {
    (runtime::DeviceAddressUtils::MallocForInput(device_context, args), ...);
  }

  template <typename... T>
  static std::pair<std::vector<kernel::KernelTensor *>, device::DeviceAddressPtrList> GetKernelTensors(
    DeviceContext *device_context, const std::vector<AbstractBasePtr> &input_abs, const T &... args) {
    std::vector<kernel::KernelTensor *> kernel_tensor_list;
    // Kernel tensor is a raw ppointer, device address need to be returned.
    device::DeviceAddressPtrList device_address_list;
    size_t index = 0;
    auto get_index = [&index]() { return index; };
    auto add_index = [&index]() { return index++; };
    (GetKernelTensor(device_context, input_abs[add_index()], get_index(), &kernel_tensor_list, &device_address_list,
                     args),
     ...);
    return std::make_pair(kernel_tensor_list, device_address_list);
  }

  static void GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const TensorPtr &tensor);

  static void GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list,
                              const std::optional<tensor::TensorPtr> &tensor);

  static void GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const std::vector<TensorPtr> &tensors);

  template <typename T>
  static void GetKernelTensor(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const T &val) {
    // Value ptr alloc device address and malloc mem here
    auto device_address = runtime::DeviceAddressUtils::CreateInputAddress(device_context, input_abs, index, val);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(device_address_list);
    MS_EXCEPTION_IF_NULL(kernel_tensor_list);
    (void)device_address_list->emplace_back(device_address);
    (void)kernel_tensor_list->emplace_back(device_address->kernel_tensor().get());
  }

  // Create input device address with kernel tensor
  template <typename... InputArgs>
  static device::DeviceAddressPtrList CreateInputDeviceAddress(DeviceContext *device_context,
                                                               const std::vector<AbstractBasePtr> &input_abs,
                                                               const InputArgs &... input_args) {
    device::DeviceAddressPtrList input_device_address;
    size_t index = 0;
    auto get_index = [&index]() { return index; };
    auto add_index = [&index]() { return index++; };
    (GetInputDeviceAddress(device_context, input_abs[add_index()], get_index(), &input_device_address, input_args),
     ...);
    return input_device_address;
  }

  template <class T>
  static void GetInputDeviceAddress(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs,
                                    size_t index, device::DeviceAddressPtrList *input_device_address, const T &t) {
    (void)input_device_address->emplace_back(
      runtime::DeviceAddressUtils::CreateInputAddress(device_context, input_abs, index, t));
  }

  static void GetInputDeviceAddress(DeviceContext *device_context, const abstract::AbstractBasePtr &input_abs,
                                    size_t index, device::DeviceAddressPtrList *input_device_address,
                                    const std::vector<tensor::TensorPtr> &t) {
    auto abs_seq = input_abs->cast<abstract::AbstractSequencePtr>();
    for (const auto &item : t) {
      (void)input_device_address->emplace_back(
        runtime::DeviceAddressUtils::CreateInputAddress(device_context, abs_seq->elements()[index], index, item));
    }
  }

  // Create output tensor device address without kernel tensor
  static void PrepareOpOutputs(DeviceContext *device_context, const std::vector<TensorPtr> &outputs) {
    runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, outputs);
  }

  // Create output tensor device address without kernel tensor
  static void MallocOpOutputs(DeviceContext *device_context, const std::vector<TensorPtr> &outputs) {
    runtime::DeviceAddressUtils::MallocForOutputs(device_context, outputs);
  }

  // Create output tensor device address with kernel tensor
  static device::DeviceAddressPtrList CreateOutputDeviceAddress(DeviceContext *device_context,
                                                                const abstract::AbstractBasePtr &abs,
                                                                const std::vector<TensorPtr> &outputs);

  // Create workspace device address with kernel tensor
  static std::vector<kernel::KernelTensor *> GetKernelTensorFromAddress(
    const device::DeviceAddressPtrList &input_device_address);

  static kernel::KernelModPtr CreateKernelMod(const PrimitivePtr &prim, const std::string &op_name,
                                              DeviceContext *device_context, const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs);
};

template <typename T>
std::vector<T> ConvertValueTupleToVector(const ValueTuplePtr &tuple) {
  std::vector<T> result;
  const auto &values = tuple->value();
  for (const auto &value : values) {
    (void)result.emplace_back(GetValue<T>(value));
  }
  MS_LOG(DEBUG) << "Convert ValueTuple to vector " << result;
  return result;
}

// Shield kernel hardware differences. Call some func of derived classes based on base classes.
// Just like SetThreadPool
class BACKEND_EXPORT PyboostKernelExtraFuncFactory {
 public:
  static PyboostKernelExtraFuncFactory &GetInstance();
  PyboostKernelExtraFuncFactory() = default;
  ~PyboostKernelExtraFuncFactory() = default;
  void AddPyboostKernelExtraFunc(const std::string &op_name, const PyboostKernelExtraFuncPtr &func) {
    kernel_func_map_[op_name] = func;
  }

  void SetThreadPool(const std::string &device_name, const kernel::KernelModPtr &kernel) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return;
    }

    iter->second->SetThreadPool(kernel);
  }

 private:
  mindspore::HashMap<std::string, PyboostKernelExtraFuncPtr> kernel_func_map_;
};

class PyboostKernelExtraFuncRegistrar {
 public:
  PyboostKernelExtraFuncRegistrar(const std::string &op_name, const PyboostKernelExtraFuncPtr &func) {
    PyboostKernelExtraFuncFactory::GetInstance().AddPyboostKernelExtraFunc(op_name, func);
  }

  ~PyboostKernelExtraFuncRegistrar() = default;
};

#define REG_PYBOOST_KERNEL_EXTRA_FUN(op_name, func) \
  static PyboostKernelExtraFuncRegistrar g_##op_name##PyboostKernelExtraFunc(#op_name, std::make_shared<func>());

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
