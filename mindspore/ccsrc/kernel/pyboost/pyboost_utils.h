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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/common/utils/tensor_future.h"
#include "runtime/pynative/op_executor.h"
#include "mindspore/core/ops/view/view_strides_calculator.h"
#include "runtime/device/device_address_utils.h"
#include "kernel/pyboost/pyboost_kernel_extra_func.h"
#include "mindspore/core/utils/simple_info.h"
#include "include/common/pynative/abstract_converter.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
using AbstractConverter = pynative::AbstractConverter;
using AddressInfoPair = std::pair<std::vector<kernel::KernelTensor *>, device::DeviceAddressPtrList>;
using BaseTensor = tensor::BaseTensor;
using BaseTensorPtr = tensor::BaseTensorPtr;
AbstractBasePtr BACKEND_EXPORT ToAbstractNoValue(const BaseTensorPtr &tensor);

class BACKEND_EXPORT PyBoostUtils {
 public:
  static AbstractBasePtr InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs);
  static void DispatchRun(const std::shared_ptr<runtime::PyBoostDeviceTask> &task);

  static DeviceSyncPtr ContiguousByDeviceAddress(const DeviceSyncPtr &device_sync);

  // Create device address
  static device::DeviceAddressPtrList CreateWorkSpaceDeviceAddress(const KernelModPtr &kernel_mod,
                                                                   const device::DeviceContext *device_context,
                                                                   const std::string &op_name);

  // Create output tensors
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::BaseTensorPtr> *outputs);
  static void CreateOutputTensor(const DeviceContext *device_context, const tensor::BaseTensorPtr &input,
                                 const TensorStorageInfoPtr &storage_info, std::vector<tensor::BaseTensorPtr> *outputs);
  static void CreateOutputTensor(const DeviceContext *device_context, const tensor::BaseTensorPtr &input,
                                 const TensorStorageInfoPtrList &storage_info_list,
                                 std::vector<tensor::BaseTensorPtr> *outputs);
  static void CreateOutputTensor(const ValueSimpleInfoPtr &output_value_simple_info,
                                 std::vector<tensor::BaseTensorPtr> *outputs);
  static void CreateOutputTensor(const TypeId &type_id, const ShapeVector &shape_vector,
                                 std::vector<tensor::BaseTensorPtr> *outputs);

  // Create input device address without kernel tensor
  template <typename... Args>
  static void PrepareOpInputs(const DeviceContext *device_context, size_t stream_id, const Args &... args) {
    size_t index = 0;
    auto add_index = [&index]() { return index++; };
    (runtime::DeviceAddressUtils::CreateInputTensorAddress(device_context, stream_id, add_index(), args), ...);
  }

  template <typename... T>
  static void MallocOpInputs(const DeviceContext *device_context, const T &... args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocInput,
                                       runtime::ProfilerRecorder::kNoName, false);
    (runtime::DeviceAddressUtils::MallocForInput(device_context, args, false), ...);
  }

  template <typename... T>
  static void MallocOpInputsForView(const DeviceContext *device_context, const T &... args) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocInput,
                                       runtime::ProfilerRecorder::kNoName, false);
    (runtime::DeviceAddressUtils::MallocForInput(device_context, args, true), ...);
  }

  template <typename... T, std::size_t... Index>
  static void GetAddressInfoHelper(const DeviceContext *device_context, size_t stream_id,
                                   const std::vector<AbstractBasePtr> &input_abs,
                                   std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                                   device::DeviceAddressPtrList *device_address_list, std::index_sequence<Index...>,
                                   const T &... args) {
    (GetKernelTensor(device_context, stream_id, input_abs[Index], Index, kernel_tensor_list, device_address_list, args),
     ...);
  }

  template <typename... T>
  static AddressInfoPair GetAddressInfo(const DeviceContext *device_context, size_t stream_id,
                                        const std::vector<AbstractBasePtr> &input_abs, const T &... args) {
    std::vector<kernel::KernelTensor *> kernel_tensor_list;
    // Kernel tensor is a raw ppointer, device address need to be returned.
    device::DeviceAddressPtrList device_address_list;
    if (input_abs.empty()) {
      std::vector<AbstractBasePtr> tmp_abs(sizeof...(args), nullptr);
      GetAddressInfoHelper(device_context, stream_id, tmp_abs, &kernel_tensor_list, &device_address_list,
                           std::make_index_sequence<sizeof...(T)>(), args...);
    } else {
      GetAddressInfoHelper(device_context, stream_id, input_abs, &kernel_tensor_list, &device_address_list,
                           std::make_index_sequence<sizeof...(T)>(), args...);
    }
    return std::make_pair(kernel_tensor_list, device_address_list);
  }

  static void LaunchKernel(const PrimitivePtr &primitive, const device::DeviceContext *device_context,
                           const AddressInfoPair &input_address_info, const AddressInfoPair &output_address_info,
                           size_t stream_id = kDefaultStreamIndex);

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const BaseTensorPtr &tensor) {
    GetKernelTensor(device_context, stream_id, nullptr, index, kernel_tensor_list, device_address_list, tensor);
  }

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const BaseTensorPtr &tensor);

  template <typename T>
  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const std::optional<T> &val) {
    if (val.has_value()) {
      GetKernelTensor(device_context, stream_id, input_abs, index, kernel_tensor_list, device_address_list,
                      val.value());
    } else {
      // Construct none kernel tensor
      MS_EXCEPTION_IF_NULL(kernel_tensor_list);
      MS_EXCEPTION_IF_NULL(device_address_list);

      const auto &kernel_tensor = std::make_shared<kernel::KernelTensor>(
        std::make_shared<abstract::TensorShape>(ShapeVector()), kTypeNone, kNone, nullptr, 0, kOpFormat_DEFAULT,
        kTypeNone->type_id(), ShapeVector(), device_context->device_context_key().device_name_,
        device_context->device_context_key().device_id_);
      kernel_tensor->set_stream_id(stream_id);
      (void)kernel_tensor_list->emplace_back(kernel_tensor.get());
      auto device_address = device_context->device_res_manager_->CreateDeviceAddress(kernel_tensor);
      (void)device_address_list->emplace_back(device_address);
    }
  }

  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list,
                              const std::vector<tensor::BaseTensorPtr> &tensors);

  template <typename T>
  static void GetKernelTensor(const DeviceContext *device_context, size_t stream_id,
                              const abstract::AbstractBasePtr &input_abs, size_t index,
                              std::vector<kernel::KernelTensor *> *kernel_tensor_list,
                              device::DeviceAddressPtrList *device_address_list, const T &val) {
    // Value ptr alloc device address and malloc mem here
    auto device_address =
      runtime::DeviceAddressUtils::CreateInputAddress(device_context, stream_id, input_abs, index, val);
    MS_EXCEPTION_IF_NULL(device_address);
    MS_EXCEPTION_IF_NULL(device_address_list);
    MS_EXCEPTION_IF_NULL(kernel_tensor_list);
    (void)device_address_list->emplace_back(device_address);
    (void)kernel_tensor_list->emplace_back(device_address->kernel_tensor().get());
  }

  // Create output tensor device address without kernel tensor
  static void PrepareOpOutputs(const DeviceContext *device_context, size_t stream_id,
                               const std::vector<tensor::BaseTensorPtr> &outputs) {
    runtime::DeviceAddressUtils::CreateOutputTensorAddress(device_context, stream_id, outputs);
  }

  // Create output tensor device address without kernel tensor
  static void MallocOpOutputs(const DeviceContext *device_context, const std::vector<tensor::BaseTensorPtr> &outputs) {
    runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kPyBoostMallocOutput,
                                       runtime::ProfilerRecorder::kNoName, false);
    runtime::DeviceAddressUtils::MallocForOutputs(device_context, outputs);
  }

  // Create workspace device address with kernel tensor
  static std::vector<kernel::KernelTensor *> GetKernelTensorFromAddress(
    const device::DeviceAddressPtrList &input_device_address);

  // Check kernel mod is reg
  static bool IsKernelModRegistered(const std::string &device_name, const std::string &op_name);
  static bool IsPyBoostCustomRegistered(const std::string &device_name, const std::string &op_name);

  static kernel::KernelModPtr CreateKernelMod(const PrimitivePtr &prim, const std::string &op_name,
                                              const DeviceContext *device_context,
                                              const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs);
  // return IsStrictlyMatched and KernelAttr
  static std::pair<bool, KernelAttr> SelectKernel(const std::vector<AbstractBasePtr> &inputs_abs,
                                                  const AbstractBasePtr &outputs_abs,
                                                  const DeviceContext *device_context, const std::string &op_name);
  static std::optional<tensor::BaseTensorPtr> CastTensor(const std::optional<tensor::BaseTensorPtr> &tensor,
                                                         const TypeId &type_id, const std::string &device_target);
  static tensor::BaseTensorPtr CastTensor(const tensor::BaseTensorPtr &tensor, const TypeId &type_id,
                                          const std::string &device_target);
  static std::vector<tensor::BaseTensorPtr> CastTensor(const std::vector<tensor::BaseTensorPtr> &tensors,
                                                       const std::vector<TypeId> &type_id_list,
                                                       const std::string &device_target);
  // ValueTuple input
  static std::vector<tensor::BaseTensorPtr> CastTensor(const std::vector<tensor::BaseTensorPtr> &tensors,
                                                       TypeId type_id, const std::string &device_target);
  template <typename... T>
  static std::pair<bool, KernelAttr> SelectKernel(AbstractConverter *converter, const DeviceContext *device_context,
                                                  const std::string &op_name,
                                                  const ValueSimpleInfoPtr &output_value_simple_info,
                                                  const T &... args) {
    // Get inputs abstract
    std::vector<AbstractBasePtr> input_abs;
    ((void)input_abs.emplace_back(converter->ConvertAbstract(args)), ...);

    // Get output abstract
    auto output_abs = TransformValueSimpleInfoToAbstract(*output_value_simple_info);
    return SelectKernel(input_abs, output_abs, device_context, op_name);
  }
  static ValueTuplePtr ConvertTensorVectorToTuple(const std::vector<BaseTensorPtr> &tensor_list) {
    vector<ValuePtr> value_vector;
    for (const auto &tensor : tensor_list) {
      (void)value_vector.emplace_back(tensor);
    }
    auto result = std::make_shared<ValueTuple>(value_vector);
    MS_LOG(DEBUG) << "Convert TensorList to ValueTuple " << result->ToString();
    return result;
  }
  static BaseTensorPtr ScalarToTensor(const ScalarPtr &scalar);

  static uint32_t cur_stream_id() { return cur_stream_id_; }

  // Set current stream for CREATE_PYBOOST_OP in front queue.
  static void set_cur_stream_id(uint32_t cur_stream_id) { cur_stream_id_ = cur_stream_id; }

 private:
  inline static uint32_t cur_stream_id_ = kDefaultStreamIndex;
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

  bool IsKernelModRegistered(const std::string &device_name, const std::string &op_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return true;
    }
    return iter->second->IsKernelModRegistered(op_name);
  }

  bool IsPyBoostCustomRegistered(const std::string &device_name, const std::string &op_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return true;
    }
    return iter->second->IsPyBoostCustomRegistered(op_name);
  }

  bool IsEnableProfiler(const std::string &device_name) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return false;
    }
    return iter->second->IsEnableProfiler();
  }

  void LaunchKernelWithProfiler(const std::string &device_name, const device::DeviceContext *device_context,
                                const std::string &op_name, const std::vector<BaseShapePtr> &base_shape,
                                const std::function<void()> &func) {
    auto iter = kernel_func_map_.find(device_name);
    if (iter == kernel_func_map_.end()) {
      return;
    }
    iter->second->LaunchKernelWithProfiler(op_name, device_context, base_shape, func);
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
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PYBOOST_UTILS_H_
