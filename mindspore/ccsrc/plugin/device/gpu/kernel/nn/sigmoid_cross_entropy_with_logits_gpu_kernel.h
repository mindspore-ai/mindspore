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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include <string>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class SigmoidCrossEntropyWithLogitsGpuKernelMod : public NativeGpuKernelMod {
 public:
  SigmoidCrossEntropyWithLogitsGpuKernelMod() = default;

  ~SigmoidCrossEntropyWithLogitsGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using SigmoidCrossEntropyWithLogitsLaunchFunc =
    std::function<bool(SigmoidCrossEntropyWithLogitsGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  SigmoidCrossEntropyWithLogitsLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, SigmoidCrossEntropyWithLogitsLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_SIGMOID_CROSS_ENTROPY_WITH_LOGITS_GPU_KERNEL_H_
