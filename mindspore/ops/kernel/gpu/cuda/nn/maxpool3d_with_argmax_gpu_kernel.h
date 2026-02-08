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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL3D_WITHARGMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL3D_WITHARGMAX_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
class MaxPool3DWithArgmaxFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit MaxPool3DWithArgmaxFwdGpuKernelMod(const std::string &kernel_name) : kernel_name_(kernel_name) {}
  ~MaxPool3DWithArgmaxFwdGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MaxPool3DArgMaxFunc =
    std::function<bool(MaxPool3DWithArgmaxFwdGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, MaxPool3DArgMaxFunc>> func_list_;
  MaxPool3DArgMaxFunc kernel_func_;
  std::string kernel_name_;
  std::vector<int64_t> ksize_;
  std::vector<int64_t> pads_;
  std::vector<int64_t> strides_;
  std::vector<int64_t> dilation_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};

  int ksize_d_{0};
  int ksize_h_{0};
  int ksize_w_{0};
  int stride_d_{0};
  int stride_h_{0};
  int stride_w_{0};
  int pad_d_{0};
  int pad_h_{0};
  int pad_w_{0};
  int dilation_d_{0};
  int dilation_h_{0};
  int dilation_w_{0};
  int in_n_{0};
  int in_c_{0};
  int in_d_{0};
  int in_h_{0};
  int in_w_{0};
  int out_d_{0};
  int out_h_{0};
  int out_w_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_MAXPOOL3D_WITHARGMAX_GPU_KERNEL_H_
