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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NON_ZERO_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NON_ZERO_CPU_KERNEL_H_

#include "plugin/device/cpu/kernel/non_zero_cpu_kernel.h"
#include <complex>
#include <vector>
#include <memory>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class NonZeroCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  NonZeroCpuKernelMod() = default;
  ~NonZeroCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 private:
  template <typename T>
  size_t NonZeroCompute(const T *input, int64_t *output, size_t input_num);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using NonZeroFunc = std::function<bool(NonZeroCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, NonZeroFunc>> func_list_;
  NonZeroFunc kernel_func_;
  CNodeWeakPtr node_wpt_;
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  size_t input_rank_{0};

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NON_ZERO_CPU_KERNEL_H_
