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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROLLING_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ROLLING_CPU_KERNEL_H_

#include <vector>
#include <string>
#include "backend/kernel_compiler/cpu/cpu_kernel.h"
#include "backend/kernel_compiler/cpu/cpu_kernel_factory.h"
#include "backend/kernel_compiler/cpu/nnacl/op_base.h"

namespace mindspore {
namespace kernel {
enum Method : int {
  Max,
  Min,
  Mean,
  Sum,
  Std,
  Var,
};
template <typename T, typename S>
class RollingCpuKernel : public CPUKernel {
 public:
  RollingCpuKernel() = default;
  ~RollingCpuKernel() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 private:
  void AxisCalculate(const std::vector<size_t> &input_shape);
  void RollingBoundsCalculate();
  void MethodSwitch();

  int64_t window_{0};
  int64_t min_periods_{0};
  int axis_{0};
  bool center_{false};
  std::string closed_{};
  Method method_{};
  std::function<S(const T *input_addr, int outer_offset, int w, int col)> reduceMethod_{};
  // shape info
  int outer_size_{0};
  int axis_size_{0};
  int inner_size_{0};
  // rolling info
  std::vector<int> starts_{};
  std::vector<int> ends_{};
};

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                      RollingCpuKernel, float, float)

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                      RollingCpuKernel, double, double)

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                      RollingCpuKernel, int32_t, int32_t)

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                      RollingCpuKernel, int64_t, int64_t)

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
                      RollingCpuKernel, int32_t, float)

MS_REG_CPU_KERNEL_T_S(Rolling, KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
                      RollingCpuKernel, int64_t, double)

}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SORT_CPU_KERNEL_H_
