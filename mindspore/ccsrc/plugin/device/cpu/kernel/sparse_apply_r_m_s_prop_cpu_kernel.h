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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_R_M_S_PORP_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_R_M_S_PORP_H_

#include <map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/sparse_optimizer_cpu_kernel.h"

namespace mindspore {
namespace kernel {
class SparseApplyRMSPropCpuKernelMod : public SparseOptimizerCpuKernelMod,
                                       public MatchKernelHelper<SparseApplyRMSPropCpuKernelMod> {
 public:
  SparseApplyRMSPropCpuKernelMod() { ResetResource(); }
  ~SparseApplyRMSPropCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  bool ResizedInputSize(const std::vector<KernelTensorPtr> &inputs);
  bool ResizedOutputSize(const std::vector<KernelTensorPtr> &outputs);
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }
  void ResetResource() noexcept;

 private:
  template <typename I, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  float rho_;
  float momentum_;
  float epsilon_;
  ShapeVector var_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_SPARSE_APPLY_R_M_S_PORP_H_
