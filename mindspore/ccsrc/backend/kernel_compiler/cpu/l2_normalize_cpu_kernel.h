/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_

#include <vector>
#include <memory>

#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class L2NormalizeCPUKernel : public CPUKernel {
 public:
  L2NormalizeCPUKernel() = default;
  ~L2NormalizeCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

  void CalcDenominator(const T *input_addr, const size_t reduce_size, const int dims,
                       std::unique_ptr<T[]> *denominator_addr);

  void CalcOutput(const T *input_addr, const std::vector<size_t> reduce_shape, const size_t output_size, T *output_addr,
                  std::unique_ptr<T[]> const &denominator_addr);

 private:
  std::vector<size_t> input_shape_;
  std::vector<size_t> output_shape_;
  T epsilon_{0};
  int axis_{0};
  void CheckParam(const CNodePtr &kernel_node);
};

MS_REG_CPU_KERNEL_T(L2Normalize, KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                    L2NormalizeCPUKernel, float16);

MS_REG_CPU_KERNEL_T(L2Normalize, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    L2NormalizeCPUKernel, float);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_
