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

#ifndef MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
#define MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_

#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "include/runtime/hardware_abstract/device_context/device_context.h"
#include "include/pynative/utils/pyboost/pyboost_kernel_extra_func.h"
#include "kernel/cpu/utils/visible.h"
#include "mindapi/base/macros.h"
#include "ir/tensor.h"
#include "ir/value.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
// Prepare inputs only for the native CPU fallback; plugin calls keep their layouts.
OPS_HOST_API bool PrepareCpuFallbackEnabled(const std::string &name);
OPS_HOST_API tensor::TensorPtr MakeCpuFallbackContiguous(const tensor::TensorPtr &input);
OPS_HOST_API tensor::TensorPtr PrepareCpuFallbackInput(const tensor::TensorPtr &input, bool *changed);
OPS_HOST_API ValuePtr PrepareCpuFallbackInput(const ValuePtr &input, bool *changed);
OPS_HOST_API ValueTuplePtr PrepareCpuFallbackInput(const ValueTuplePtr &input, bool *changed);

template <typename T>
T PrepareCpuFallbackInput(const T &input, bool *) {
  return input;
}

template <typename T>
std::optional<T> PrepareCpuFallbackInput(const std::optional<T> &input, bool *changed) {
  if (!input.has_value()) {
    return input;
  }
  return PrepareCpuFallbackInput(input.value(), changed);
}

template <typename... Args>
auto PrepareCpuFallbackInputs(const std::tuple<Args...> &inputs, bool *changed) {
  return std::apply(
    [changed](const auto &...args) { return std::make_tuple(PrepareCpuFallbackInput(args, changed)...); }, inputs);
}

class OPS_HOST_API PyboostCPUKernelExtraFunc : public PyboostKernelExtraFunc {
 public:
  void SetThreadPool(const kernel::KernelModPtr &kernel) override;
  bool IsKernelModRegistered(const std::string &op_name) override;
  bool IsPyBoostCustomRegistered(const std::string &op_name) override;
  bool IsEnableProfiler() override;
  void LaunchKernelWithProfiler(const std::string &op_name, const device::DeviceContext *device_context,
                                const std::vector<BaseShapePtr> &base_shape,
                                const std::function<void()> &func) override;
};
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_PYBOOST_CPU_KERNRL_EXTRA_FUNC_H_
