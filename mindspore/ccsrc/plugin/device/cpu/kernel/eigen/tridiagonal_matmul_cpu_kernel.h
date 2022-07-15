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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_MATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_MATMUL_CPU_KERNEL_H_
#include <complex>
#include <memory>
#include <unordered_map>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/backend/common/session/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
class TridiagonalMatMulCpuKernelMod : public DeprecatedNativeCpuKernelMod {
 public:
  TridiagonalMatMulCpuKernelMod() = default;
  ~TridiagonalMatMulCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  CNodeWeakPtr node_wpt_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void LaunchTridiagonalMatMul(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_MATMUL_CPU_KERNEL_H_
