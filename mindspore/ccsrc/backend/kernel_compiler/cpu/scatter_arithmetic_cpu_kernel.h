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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ARITHMETIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ARITHMETIC_CPU_KERNEL_H_

#include <vector>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
template <typename T>
class ScatterArithmeticCPUKernel : public CPUKernel {
 public:
  ScatterArithmeticCPUKernel() = default;

  ~ScatterArithmeticCPUKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void InitComputeFunc();
  void ScatterAdd(T *input, const int *indices, const T *updates) const;
  void ScatterSub(T *input, const int *indices, const T *updates) const;
  void ScatterMul(T *input, const int *indices, const T *updates) const;
  void ScatterDiv(T *input, const int *indices, const T *updates) const;
  void ScatterMax(T *input, const int *indices, const T *updates) const;
  void ScatterMin(T *input, const int *indices, const T *updates) const;
  void ScatterUpdate(T *input, const int *indices, const T *updates) const;

  using TypeComputeFunc = std::function<void(ScatterArithmeticCPUKernel *, T *, const int *, const T *)>;

  TypeComputeFunc compute_func_;
  size_t input_size_{0};
  size_t inner_size_{0};
  size_t indices_size_{0};
  const size_t INPUT_INDEX_{0};
  const size_t INDICES_INDEX_{1};
  const size_t UPDATES_INDEX_{2};
  const size_t OUTPUT_INDEX_{0};
};

MS_REG_CPU_KERNEL_T(ScatterAdd,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterAdd,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterAdd,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterSub,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterSub,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterSub,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterMul,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterMul,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterMul,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterDiv,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterDiv,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterDiv,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterMax,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterMax,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterMax,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterMin,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterMin,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterMin,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
MS_REG_CPU_KERNEL_T(ScatterUpdate,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddOutputAttr(kNumberTypeInt32),
                    ScatterArithmeticCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ScatterUpdate,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeFloat32)
                      .AddOutputAttr(kNumberTypeFloat32),
                    ScatterArithmeticCPUKernel, float);
MS_REG_CPU_KERNEL_T(ScatterUpdate,
                    KernelAttr()
                      .AddInputAttr(kNumberTypeInt64)
                      .AddInputAttr(kNumberTypeInt32)
                      .AddInputAttr(kNumberTypeInt64)
                      .AddOutputAttr(kNumberTypeInt64),
                    ScatterArithmeticCPUKernel, int64_t);
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SCATTER_ARITHMETIC_CPU_KERNEL_H_
