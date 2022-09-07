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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LINEAR_SUM_ASSIGNMENT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LINEAR_SUM_ASSIGNMENT_CPU_KERNEL_H_

#include <vector>
#include <complex>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class LinearSumAssignmentCpuKernelMod : public NativeCpuKernelMod,
                                        public MatchKernelHelper<LinearSumAssignmentCpuKernelMod> {
 public:
  LinearSumAssignmentCpuKernelMod() = default;
  ~LinearSumAssignmentCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  std::vector<int64_t> cost_matrix_shape_;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs);

  template <typename T>
  int64_t AugmentingPath(int64_t nc, const T *cost, std::vector<T> *u, std::vector<T> *v, std::vector<int64_t> *path,
                         std::vector<int64_t> *row4col, std::vector<T> *shortest_path_costs, int64_t i,
                         std::vector<bool> *SR, std::vector<bool> *SC, std::vector<int64_t> *remaining,
                         T *p_min_val) const;

  template <typename T>
  bool Solve(int64_t nr, int64_t nc, int64_t raw_rc, T *cost, bool maximize, int64_t *a, int64_t *b) const;

  template <typename T>
  void ReArrange(int64_t *origin_nr, int64_t *origin_nc, int64_t raw_nc, std::vector<T> *temp, const T *cost,
                 bool transpose, bool maximize) const;

  void PostProcess(int64_t *a, int64_t *b, const std::vector<int64_t> &col4row, bool transpose, int64_t nr, int64_t nc,
                   int64_t element_num) const;

  void AugmentPreviousSolution(int64_t j, int64_t cur_row, std::vector<int64_t> *path, std::vector<int64_t> *row4col,
                               std::vector<int64_t> *col4row) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LINEAR_SUM_ASSIGNMENT_CPU_KERNEL_H_
