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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LUSOLVE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LUSOLVE_CPU_KERNEL_H_
#include <Eigen/Dense>
#include <vector>
#include <memory>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/cpu_kernel_factory.h"

namespace mindspore {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
namespace kernel {
template <typename T1, typename T2>
class LuSolveCpuKernelMod : public NativeCpuKernelMod {
 public:
  LuSolveCpuKernelMod() = default;
  ~LuSolveCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;
  void LuSolve(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
               T1 *b_working_ptr, T1 *lu_working_ptr, int32_t *pivots_working_ptr, size_t b_stride, size_t a);

 private:
  CNodePtr node_wpt_;
  std::vector<size_t> input_0_Shape;
  std::vector<size_t> input_1_Shape;
  std::vector<size_t> input_2_Shape;
};

MS_REG_CPU_KERNEL_T_S(LuSolve,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeFloat16)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat16),
                      LuSolveCpuKernelMod, float, float16);
MS_REG_CPU_KERNEL_T_S(LuSolve,
                      KernelAttr()
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeFloat32)
                        .AddInputAttr(kNumberTypeInt32)
                        .AddOutputAttr(kNumberTypeFloat32),
                      LuSolveCpuKernelMod, float, float);
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LUSOLVE_CPU_KERNEL_H_
