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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RELU_V2_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RELU_V2_GPU_KERNEL_H_

#include <vector>
#include <algorithm>
#include <functional>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ReluV2GpuKernelMod : public NativeGpuKernelMod {
 public:
  ReluV2GpuKernelMod() = default;
  ~ReluV2GpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  using ReLUV2FwLaunchFunc =
    std::function<bool(ReluV2GpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  std::string kernel_name_{};
  ReLUV2FwLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, ReLUV2FwLaunchFunc>> func_list_;
  size_t element_num_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RELU_V2_GPU_KERNEL_H_
