/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_

#include <utility>
#include <map>
#include <iostream>
#include <vector>
#include <string>

#include "ops/grad/mirror_pad_grad.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/mirror_pad_impl.cuh"

namespace mindspore {
namespace kernel {
class MirrorPadGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  MirrorPadGradGpuKernelMod() = default;
  ~MirrorPadGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using MirrorPadGradLaunchFunc =
    std::function<bool(MirrorPadGradGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

 private:
  void CalculateWorkspace(const ShapeVector &input_shape, const std::vector<size_t> &output_shape);

  MirrorPadGradLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, MirrorPadGradLaunchFunc>> func_list_;

  size_t num_input_{0};
  int num_paddings_{0};
  int mode_{0};
  bool is_null_input_{false};
  std::string kernel_name_{"MirrorPadGrad"};
  std::vector<int> input_shape_;
  std::vector<int> output_shape_;
  size_t input_size_{1};
  size_t output_size_{1};
  size_t workspace_size_{0};
  size_t input_type_size_{0};
  size_t padding_type_size_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_MIRROR_PAD_GRAD_GPU_KERNEL_H_
