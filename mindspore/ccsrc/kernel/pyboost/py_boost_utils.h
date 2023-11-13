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

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT PyBoostUtils {
 public:
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs);
  static void CreateOutputTensor(const AbstractBasePtr &abstract, std::vector<tensor::TensorPtr> *outputs,
                                 std::vector<pynative::DeviceAddressPromisePtr> *device_sync_promises);
  static DeviceContext *GetDeviceContext(const std::string &device_type);
  static void CreateOutputTensor(const tensor::TensorPtr &input, const TensorStorageInfoPtr &storage_info,
                                 std::vector<tensor::TensorPtr> *outputs);
  static AbstractBasePtr InferByOpDef(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_abs);
};

DeviceContext *CreateOrGetDeviceContextAndInit(const std::string &target_device);
tensor::TensorPtr BACKEND_EXPORT ScalarToTensor(const ScalarPtr &scalar, const TypePtr &type);
tensor::TensorPtr BACKEND_EXPORT ContiguousTensor(const tensor::TensorPtr &input_tensor);

template <typename... Args>
void PrepareOpInputs(DeviceContext *device_context, Args... args) {
  [&]() { (runtime::DeviceAddressUtils::CreateInputAddress(device_context, args, "input"), ...); }();
}

std::vector<kernel::KernelTensor *> BACKEND_EXPORT GetWorkspaceKernelTensors(
  const KernelModPtr &kernel_mod, const device::DeviceContext *device_context, const std::string &op_name);

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

std::vector<device::DeviceAddressPtr> BACKEND_EXPORT PrepareOpOutputs(DeviceContext *device_context,
                                                                      const std::vector<TensorPtr> &outputs);
std::vector<device::DeviceAddressPtr> BACKEND_EXPORT
PrepareOpOutputs(DeviceContext *device_context, const std::vector<TensorPtr> &outputs,
                 const std::vector<pynative::DeviceAddressPromisePtr> &device_sync_promises);
void BACKEND_EXPORT DispatchRun(const std::shared_ptr<pynative::PyBoostDeviceTask> &task);
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_PY_BOOST_UTILS_H_
