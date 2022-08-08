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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include "mindspore/core/ops/max_pool_with_argmax.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class MaxPoolWithArgmaxCpuKernelMod : public NativeCpuKernelMod {
 public:
  MaxPoolWithArgmaxCpuKernelMod() {}
  ~MaxPoolWithArgmaxCpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;
  void ResizedInputSize(const std::vector<KernelTensorPtr> &inputs);
  void ResizedOutputSize(const std::vector<KernelTensorPtr> &outputs);
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);

  using MaxPoolWithArgmaxFunc = std::function<bool(
    MaxPoolWithArgmaxCpuKernelMod *, const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &)>;

  static std::vector<std::pair<KernelAttr, MaxPoolWithArgmaxFunc>> func_list_;
  MaxPoolWithArgmaxFunc kernel_func_;
  int batch_ = 0;
  int channel_ = 0;
  int input_height_ = 0;
  int input_width_ = 0;
  int window_height_ = 1;
  int window_width_ = 1;
  int stride_height_ = 1;
  int stride_width_ = 1;
  PadMode pad_mode_ = PadMode::VALID;
  Format data_format_ = Format::NCHW;
  int pad_height_ = 0;
  int pad_width_ = 0;
  int pad_top_ = 0;
  int pad_left_ = 0;
  int output_height_ = 0;
  int output_width_ = 0;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MAX_POOL_WITH_ARGMAX_CPU_KERNEL_H_
