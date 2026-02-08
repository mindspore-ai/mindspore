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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_

#include <map>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ResizeBilinearGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeBilinearGpuKernelMod() = default;
  ~ResizeBilinearGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex1}; }

  using ResizeBilinearFunc =
    std::function<bool(ResizeBilinearGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;

 private:
  float Scaling(const int in_size, const int out_size, bool align_corners) {
    if (align_corners && out_size > 1) {
      return (in_size - 1) / static_cast<float>(out_size - 1);
    } else {
      if (out_size == 0) {
        MS_EXCEPTION(ValueError) << "For " << kernel_name_ << ", out_size cannot be 0, but got out_size = " << out_size
                                 << ", align_corners = " << align_corners << ".";
      }
      return in_size / static_cast<float>(out_size);
    }
  }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  static std::vector<std::pair<KernelAttr, ResizeBilinearFunc>> func_list_;
  ResizeBilinearFunc kernel_func_;

  bool align_corners_;
  bool half_pixel_centers_;
  bool is_null_input_;
  int n_;
  int c_;
  int input_h_;
  int input_w_;
  int output_h_;
  int output_w_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_RESIZE_BILINEAR_GPU_KERNEL_H_
