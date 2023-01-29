/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LPNORM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LPNORM_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <string>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class LpNormCpuKernelMod : public NativeCpuKernelMod {
 public:
  LpNormCpuKernelMod() = default;
  ~LpNormCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return is_null_input_ ? true : kernel_func_(this, inputs, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool GetReductionAttr(const BaseOperatorPtr &base_operator);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using LpNromFunc = std::function<bool(LpNormCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                        const std::vector<kernel::AddressPtr> &)>;
  LpNromFunc kernel_func_;
  bool is_p_zero_{false};
  bool is_null_input_{false};
  bool is_scalar_input_{false};
  size_t input_elements_{1};
  size_t reduce_size_{1};
  float p_{2.0};
  float epsilon_{1e-12};
  std::vector<int64_t> axis_;
  std::vector<size_t> physical_indexes_;
  static std::vector<std::pair<KernelAttr, LpNromFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LPNORM_CPU_KERNEL_H_
