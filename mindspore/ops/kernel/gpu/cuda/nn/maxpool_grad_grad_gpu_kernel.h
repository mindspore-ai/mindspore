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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL_GRAD_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL_GRAD_GRAD_GPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
constexpr int kMaxPool2DGradGradDim = 2;
constexpr int kMaxPool3DGradGradDim = 3;

class MaxPoolGradGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit MaxPoolGradGradGpuKernelMod(const int &dim) : dim_(dim) {}
  ~MaxPoolGradGradGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  using MaxPoolGradGradFunc =
    std::function<bool(MaxPoolGradGradGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;

  void *cuda_stream_{nullptr};
  MaxPoolGradGradFunc kernel_func_{};
  static std::vector<std::pair<KernelAttr, MaxPoolGradGradFunc>> func_list_;

  void CalPad();

  std::vector<int64_t> kernels_;
  std::vector<int64_t> strides_;
  PadMode pad_mode_;
  std::vector<int64_t> in_shapes_;
  std::vector<int64_t> out_shapes_;

  int dim_ = 0;
  int batch_ = 0;
  int channel_ = 0;
  int input_depth_ = 0;
  int input_height_ = 0;
  int input_width_ = 0;
  int output_depth_ = 0;
  int output_height_ = 0;
  int output_width_ = 0;

  int window_depth_ = 0;
  int window_height_ = 0;
  int window_width_ = 0;
  int stride_depth_ = 0;
  int stride_height_ = 0;
  int stride_width_ = 0;
  int pad_front_ = 0;
  int pad_top_ = 0;
  int pad_left_ = 0;

  size_t depth_index_ = 0;
  size_t height_index_ = 0;
  size_t width_index_ = 0;
  size_t input_batch_stride_ = 0;
  size_t output_batch_stride_ = 0;
};

class MaxPool2DGradGradGpuKernelMod : public MaxPoolGradGradGpuKernelMod {
 public:
  MaxPool2DGradGradGpuKernelMod() : MaxPoolGradGradGpuKernelMod(kMaxPool2DGradGradDim) {}
  ~MaxPool2DGradGradGpuKernelMod() = default;
};

class MaxPool3DGradGradGpuKernelMod : public MaxPoolGradGradGpuKernelMod {
 public:
  MaxPool3DGradGradGpuKernelMod() : MaxPoolGradGradGpuKernelMod(kMaxPool3DGradGradDim) {}
  ~MaxPool3DGradGradGpuKernelMod() = default;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL_GRAD_GRAD_GPU_KERNEL_H_
