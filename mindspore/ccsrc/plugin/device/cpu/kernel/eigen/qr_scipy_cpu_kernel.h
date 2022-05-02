/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_QR_SCIPY_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_QR_SCIPY_CPU_KERNEL_H_

#include <utility>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class QRCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  QRCpuKernelMod() = default;
  ~QRCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using QRFunc = std::function<bool(QRCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, QRFunc>> func_list_;
  QRFunc kernel_func_;

  size_t a_row_{0};
  size_t a_col_{0};
  size_t q_row_{0};
  size_t q_col_{0};
  size_t r_row_{0};
  size_t r_col_{0};
  bool economic_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_QR_SCIPY_CPU_KERNEL_H_
