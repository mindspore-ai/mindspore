/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class BACKEND_EXPORT FusedSparseFtrlCpuKernelMod : public SparseOptimizerCpuKernelMod,
                                                   public MatchKernelHelper<FusedSparseFtrlCpuKernelMod> {
 public:
  FusedSparseFtrlCpuKernelMod() = default;
  ~FusedSparseFtrlCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  void ResetResource() noexcept;

  float lr_{0.0};
  float l1_{0.0};
  float l2_{0.0};
  float lr_power_{0.0};

 private:
  template <typename T>
  void InitWorkspaceSize();

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &) const;
};

class BACKEND_EXPORT SparseApplyFtrlCpuKernelMod : public SparseOptimizerCpuKernelMod,
                                                   public MatchKernelHelper<SparseApplyFtrlCpuKernelMod> {
 public:
  SparseApplyFtrlCpuKernelMod() = default;
  ~SparseApplyFtrlCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  void ResetResource() noexcept;

  float lr_{0.0};
  float l1_{0.0};
  float l2_{0.0};
  float lr_power_{0.0};

 private:
  int64_t batch_rank_{0};
  int64_t batch_size_{1};
  size_t indices_inner_size_{0};
  size_t grad_inner_size_{0};
  size_t var_inner_size_{0};
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &) const;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_FTRL_CPU_KERNEL_H_
