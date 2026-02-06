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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RESIZE_LINEAR_1D_GRAD_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RESIZE_LINEAR_1D_GRAD_GPU_KERNEL_H

#include <vector>
#include <algorithm>
#include <string>
#include <map>
#include <utility>
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/resize_linear_1d.cuh"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "mindspore/ops/infer/ops_func_impl/resize_linear_1d.h"
#include "include/runtime/hardware_abstract/kernel_base/common_utils.h"

namespace mindspore {
namespace kernel {
constexpr auto kUnKnown = "UnKnown";
constexpr auto kResizeLinear1DGrad = "ResizeLinear1DGrad";
class ResizeLinear1DGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeLinear1DGradGpuKernelMod() {}
  ~ResizeLinear1DGradGpuKernelMod() {}

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);

  using ResizeLinear1DGradFunc = std::function<bool(
    ResizeLinear1DGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;

 private:
  ResizeLinear1DGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, ResizeLinear1DGradFunc>> func_list_;

  std::vector<int64_t> grad_output_shape_;
  std::vector<int64_t> grad_input_shape_;

  int64_t batch_{0};
  int64_t channel_{0};
  int64_t in_width_{0};
  int64_t out_width_{0};
  ResizeLinearCoordinateTransformationMode mode_{ResizeLinearCoordinateTransformationMode::ALIGN_CORNERS};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_RESIZE_LINEAR_1D_GRAD_GPU_KERNEL_H
