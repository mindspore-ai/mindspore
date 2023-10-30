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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include "mindspore/core/ops/grad/max_pool_grad_with_argmax.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/op_utils.h"

namespace mindspore {
namespace kernel {
class MaxPoolGradWithArgmaxCpuKernelMod : public NativeCpuKernelMod {
 public:
  MaxPoolGradWithArgmaxCpuKernelMod() {}
  ~MaxPoolGradWithArgmaxCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  void ResizedInputSize(const std::vector<KernelTensor *> &inputs);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using MaxPoolGradWithArgmaxFunc =
    std::function<bool(MaxPoolGradWithArgmaxCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  static std::vector<std::pair<KernelAttr, MaxPoolGradWithArgmaxFunc>> func_list_;
  MaxPoolGradWithArgmaxFunc kernel_func_;
  int batch_ = 0;
  int channel_ = 0;
  int x_height_ = 0;
  int x_width_ = 0;
  int dy_height_ = 0;
  int dy_width_ = 0;
  int stride_height_ = 1;
  int stride_width_ = 1;
  PadMode pad_mode_ = PadMode::VALID;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_GRAD_WITH_ARGMAX_CPU_KERNEL_H_
