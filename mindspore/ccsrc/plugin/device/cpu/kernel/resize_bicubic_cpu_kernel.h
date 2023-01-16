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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_BICUBIC_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_BICUBIC_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class ResizeBicubicCPUKernelMod : public NativeCpuKernelMod {
 public:
  ResizeBicubicCPUKernelMod() = default;
  ~ResizeBicubicCPUKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  void CalResizeClass(const CNodePtr &kernel_node, const std::vector<AddressPtr> &inputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  template <typename T1, typename T2>
  inline void interpolate_with_caching(const T1 *input_data, const bool half_pixel_centers_, T2 *output_data);

  using ResizeBicubicFunc = std::function<bool(ResizeBicubicCPUKernelMod *, const std::vector<kernel::AddressPtr> &,
                                               const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, ResizeBicubicFunc>> func_list_;
  ResizeBicubicFunc kernel_func_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_RESIZE_BICUBIC_CPU_KERNEL_H_
