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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_LU_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_LU_CPU_KERNEL_H_

#include <map>
#include <utility>
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
  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  T GetPermutatedValue(const T *lu_value, const std::vector<int> &per_value, size_t i, size_t j) const;
  template <typename T>
  bool UpdateMajorPermutation(T *lu_value, std::vector<int> *per_value, int *pivots, size_t k, size_t rows) const;
  template <typename T>
  void SetPermutatedValue(T *lu_value, const std::vector<int> &per_value, size_t i, size_t j, const T &value) const;
  void DoSafeMemCopy(void *dest, size_t dest_max, const void *src, size_t count) const;
  template <typename T>
  void InitIOSize(const CNodePtr &kernel_node);
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);
  using LUFunc = std::function<bool(LUCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                    const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, LUFunc>> func_list_;
  LUFunc kernel_func_;

  void InitMatrixInfo(const std::vector<size_t> &shape, size_t *row, size_t *col);
  void InitPivotVecInfo(const std::vector<size_t> &shape, size_t *row, size_t *col) const;
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

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_EIGEN_LU_CPU_KERNEL_H_
