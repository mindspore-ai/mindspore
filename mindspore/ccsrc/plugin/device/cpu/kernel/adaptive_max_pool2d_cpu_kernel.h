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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAXPOOL2D_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAXPOOL2D_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
static inline size_t start_index(size_t dim, size_t output_range, size_t input_range) {
  return static_cast<size_t>(std::floor(dim * input_range / static_cast<float>(output_range)));
}

static inline size_t end_index(size_t dim, size_t output_range, size_t input_range) {
  return static_cast<size_t>(std::ceil((dim + 1) * input_range / static_cast<float>(output_range)));
}

class AdaptiveMaxPool2dCpuKernelMod : public NativeCpuKernelMod,
                                      public MatchKernelHelper<AdaptiveMaxPool2dCpuKernelMod> {
 public:
  AdaptiveMaxPool2dCpuKernelMod() = default;
  ~AdaptiveMaxPool2dCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool ResizedInputSize(const std::vector<KernelTensorPtr> &inputs);

  bool ResizedOutputSize();

  bool UpdateOutputSizeList(const std::vector<KernelTensorPtr> &outputs, size_t input_type_size);

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &,
                    const std::vector<kernel::AddressPtr> &outputs);

  using FuncList = std::vector<std::pair<KernelAttr, AdaptiveMaxPool2dCpuKernelMod::KernelRunFunc>>;

  size_t input_height_{0};
  size_t input_width_{0};
  size_t output_height_{0};
  size_t output_width_{0};
  // The number of N * C.
  size_t channel_size_{0};
  // The number of H * W.
  size_t input_hw_{0};
  size_t output_hw_{0};
  std::vector<int64_t> attr_output_size_;
  bool attr_return_indices_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ADAPTIVE_MAXPOOL2D_CPU_KERNEL_H_
