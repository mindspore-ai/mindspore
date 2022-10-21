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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_

#include <map>
#include <string>
#include <utility>
#include <vector>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"

namespace mindspore {
namespace kernel {
class ResizeBilinearGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  ResizeBilinearGradGpuKernelMod() = default;
  ~ResizeBilinearGradGpuKernelMod() override = default;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  std::vector<KernelAttr> GetOpSupport() override;

  float Scaling(const int in_size, const int out_size, bool align_corners) {
    return (align_corners && out_size > 1) ? (in_size - 1) / static_cast<float>(out_size - 1)
                                           : in_size / static_cast<float>(out_size);
  }
  using ResizeBilinearGradFunc =
    std::function<bool(ResizeBilinearGradGpuKernelMod *, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, const std::vector<AddressPtr> &, void *)>;

 private:
  void ResetResource() noexcept {
    align_corners_ = false;
    half_pixel_centers_ = false;
    is_null_input_ = false;
    n_ = 0;
    c_ = 0;
    dy_h_ = 0;
    dy_w_ = 0;
    dx_h_ = 0;
    dx_w_ = 0;
    dy_size_ = 0;
    dx_size_ = 0;
    workspace_size_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() { workspace_size_list_.push_back(workspace_size_); }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  template <typename T>
  bool LaunchHalfKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                        const std::vector<AddressPtr> &outputs, void *stream_ptr);
  static std::vector<std::pair<KernelAttr, ResizeBilinearGradFunc>> func_list_;
  ResizeBilinearGradFunc kernel_func_;

  bool align_corners_;
  bool half_pixel_centers_;
  bool is_null_input_;
  int n_;
  int c_;
  int dy_h_;
  int dy_w_;
  int dx_h_;
  int dx_w_;
  size_t dy_size_;
  size_t dx_size_;
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_RESIZE_BILINEAR_GRAD_GPU_KERNEL_H_
