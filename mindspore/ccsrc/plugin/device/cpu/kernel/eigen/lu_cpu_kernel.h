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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_

#include <tuple>
#include <vector>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class LUCpuKernelMod : public NativeCpuKernelMod {
 public:
  LUCpuKernelMod() = default;
  ~LUCpuKernelMod() override = default;
  void InitKernel(const CNodePtr &kernel_node) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  T GetPermutatedValue(const T *lu_value, const std::vector<int> &per_value, size_t i, size_t j);
  template <typename T>
  bool UpdateMajorPermutation(T *lu_value, std::vector<int> *per_value, int *pivots, size_t k, size_t rows);
  template <typename T>
  void SetPermutatedValue(T *lu_value, const std::vector<int> &per_value, size_t i, size_t j, const T &value);
  template <typename T>
  void InitIOSize(const CNodePtr &kernel_node);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using LUFunc = std::function<bool(LUCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  using InitFunc = std::function<void(LUCpuKernelMod *, const CNodePtr &)>;
  static std::vector<std::tuple<KernelAttr, LUFunc, InitFunc>> func_list_;
  LUFunc kernel_func_;
  InitFunc init_io_func_;

  void InitInputOutputSize(const CNodePtr &kernel_node) override { init_io_func_(this, kernel_node); }

  void InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);
  void InitPivotVecInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);
  size_t batch_size_{1};
  size_t a_row_{1};
  size_t a_col_{1};
  size_t lu_row_{1};
  size_t lu_col_{1};
  size_t pivots_row_{1};
  size_t pivots_col_{1};
  size_t permutation_row_{1};
  size_t permutation_col_{1};
  TypeId dtype_{kNumberTypeFloat32};
  int *batch_pivots_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_LU_CPU_KERNEL_H_
