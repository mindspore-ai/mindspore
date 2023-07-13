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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <utility>
#include <vector>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class UpsampleNearest3DCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpsampleNearest3DCpuKernelMod() = default;
  ~UpsampleNearest3DCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

 private:
  void ComputeNearestIndex(int64_t *const indices, const int64_t stride, const int64_t input_szie,
                           const int64_t output_size, const double scale) const;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);
  using KernelRunFunc = std::function<bool(UpsampleNearest3DCpuKernelMod *, const std::vector<AddressPtr> &,
                                           const std::vector<AddressPtr> &, const std::vector<AddressPtr> &)>;
  KernelRunFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list_;
  size_t unit_size_{0};
  std::vector<int64_t> x_shape_;
  std::vector<int64_t> y_shape_;
  std::vector<double> scales_;
  std::vector<int64_t> none_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_CPU_KERNEL_H_
