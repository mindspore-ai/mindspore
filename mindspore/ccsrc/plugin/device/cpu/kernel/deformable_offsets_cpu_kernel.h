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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class DeformableOffsetsCpuKernelMod : public NativeCpuKernelMod,
                                      public MatchKernelHelper<DeformableOffsetsCpuKernelMod> {
 public:
  DeformableOffsetsCpuKernelMod() { ResetResource(); }
  ~DeformableOffsetsCpuKernelMod() override = default;

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

 private:
  void ResetResource() noexcept;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &workspace,
                    const std::vector<kernel::AddressPtr> &outputs);

  void GenPositionGrid(int64_t *position_grid);

  template <typename T>
  void DeformableOffsets(const T *x_addr, const T *offsets_addr, const int64_t *position_grid_addr, T *output_addr);

  std::vector<int64_t> strides_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> dilations_;
  int64_t deformable_groups_{1};
  bool modulated_{true};

  size_t n_axis_{kIndex0};
  size_t c_axis_{kIndex1};
  size_t h_axis_{kIndex2};
  size_t w_axis_{kIndex3};
  int64_t n_{0};
  int64_t c_{0};
  int64_t input_h_{0};
  int64_t input_w_{0};
  int64_t output_h_{0};
  int64_t output_w_{0};
  int64_t position_grid_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_DEFORMABLE_OFFSETS_CPU_KERNEL_H_
