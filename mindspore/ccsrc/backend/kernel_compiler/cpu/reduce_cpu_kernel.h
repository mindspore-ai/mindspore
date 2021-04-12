/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
enum class ReduceType { ReduceAll, ReduceAny, ReduceMax, ReduceMin, ReduceSum, ReduceMean };

template <typename T>
class ReduceCPUKernel : public CPUKernel {
 public:
  ReduceCPUKernel() = default;
  ~ReduceCPUKernel() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  std::vector<size_t> input_shape_;
  std::vector<int64_t> axis_;
  ReduceType reduce_type_;
  std::function<void(const T *, size_t, T *)> reduce_func_;
};

MS_REG_CPU_KERNEL_T(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    ReduceCPUKernel, float);
MS_REG_CPU_KERNEL_T(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    ReduceCPUKernel, double);
MS_REG_CPU_KERNEL_T(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    ReduceCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ReduceMean, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    ReduceCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    ReduceCPUKernel, float);
MS_REG_CPU_KERNEL_T(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    ReduceCPUKernel, double);
MS_REG_CPU_KERNEL_T(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    ReduceCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ReduceMax, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    ReduceCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    ReduceCPUKernel, float);
MS_REG_CPU_KERNEL_T(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    ReduceCPUKernel, double);
MS_REG_CPU_KERNEL_T(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    ReduceCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ReduceSum, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    ReduceCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                    ReduceCPUKernel, float);
MS_REG_CPU_KERNEL_T(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                    ReduceCPUKernel, double);
MS_REG_CPU_KERNEL_T(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                    ReduceCPUKernel, int32_t);
MS_REG_CPU_KERNEL_T(ReduceMin, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                    ReduceCPUKernel, int64_t);

MS_REG_CPU_KERNEL_T(ReduceAll, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    ReduceCPUKernel, bool);

MS_REG_CPU_KERNEL_T(ReduceAny, KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
                    ReduceCPUKernel, bool);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_REDUCE_CPU_KERNEL_H_
