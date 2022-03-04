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
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
constexpr size_t kInputNum = 3;
constexpr size_t kOutputNum = 1;
namespace kernel {
class LuSolveCpuKernelMod : public NativeCpuKernelMod {
 public:
  LuSolveCpuKernelMod() = default;
  ~LuSolveCpuKernelMod() override = default;

  void InitKernel(const CNodePtr &kernel_node) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T1, typename T2>
  void LuSolve(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs,
               T1 *b_working_ptr, T1 *lu_working_ptr, int32_t *pivots_working_ptr, size_t b_stride, size_t a);
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using LuSolveFunc = std::function<bool(LuSolveCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, LuSolveFunc>> func_list_;
  LuSolveFunc kernel_func_;
  CNodePtr node_wpt_;
  std::vector<size_t> input_0_Shape;
  std::vector<size_t> input_1_Shape;
  std::vector<size_t> input_2_Shape;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LUSOLVE_CPU_KERNEL_H_
