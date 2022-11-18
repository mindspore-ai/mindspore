/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <utility>
#include <map>
#include <tuple>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class PrintCpuKernelMod : public NativeCpuKernelMod {
 public:
  PrintCpuKernelMod() = default;
  ~PrintCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchKernel(size_t index, const std::vector<kernel::AddressPtr> &inputs);
  using PrintFunc = std::function<void(PrintCpuKernelMod *, size_t, const std::vector<kernel::AddressPtr> &)>;
  static std::map<TypeId, PrintCpuKernelMod::PrintFunc> func_map_;
  PrintFunc kernel_func_;

  std::vector<ShapeVector> input_shapes_;
  std::vector<size_t> input_sizes_;
  std::vector<TypeId> data_types_;

  std::unordered_map<int64_t, int64_t> value_type_;
  std::vector<std::tuple<size_t, TypeId>> input_info_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PRINT_CPU_KERNEL_H_
