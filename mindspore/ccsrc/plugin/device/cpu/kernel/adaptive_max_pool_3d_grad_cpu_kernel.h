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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_GRAD_CPU_KERNEL_H_

#include <Eigen/Dense>
#include <map>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <vector>
#include <memory>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class AdaptiveMaxPool3DGradCpuKernelMod : public NativeCpuKernelMod,
                                          public MatchKernelHelper<AdaptiveMaxPool3DGradCpuKernelMod> {
 public:
  AdaptiveMaxPool3DGradCpuKernelMod() = default;
  ~AdaptiveMaxPool3DGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  };
  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;
  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return MatchKernelHelper::OpSupport(); }

 private:
  std::vector<int64_t> input_grad_shape_;
  std::vector<int64_t> input_x_shape_;
  std::vector<int64_t> input_argmax_shape_;
  std::vector<int64_t> output_shape_;
  TypeId input_grad_dtype;
  TypeId input_x_dtype;
  TypeId input_argmax_dtype;
  TypeId output_dtype;
  const size_t kInputNum = 3;
  const size_t kOutputNum = 1;

  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  bool LaunchKernelHalf(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                        const std::vector<kernel::AddressPtr> &outputs);
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAX_POOL_3D_GRAD_CPU_KERNEL_H_
