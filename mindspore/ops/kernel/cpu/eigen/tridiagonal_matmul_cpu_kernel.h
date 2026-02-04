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
#include <map>
#include <vector>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"
#include "mindspore/ccsrc/include/backend/common/kernel_graph/anf_runtime_algorithm.h"

namespace mindspore {
namespace kernel {
class TridiagonalMatMulCpuKernelMod : public NativeCpuKernelMod {
 public:
  TridiagonalMatMulCpuKernelMod() = default;
  ~TridiagonalMatMulCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  ShapeVector rhs_shape_;
  TypeId dtype_{kTypeUnknown};
  template <typename T>
  void LaunchTridiagonalMatMul(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_MATMUL_CPU_KERNEL_H_
