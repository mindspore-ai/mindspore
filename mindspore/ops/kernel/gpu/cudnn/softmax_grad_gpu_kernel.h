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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_

#include <algorithm>
#include <functional>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
namespace mindspore {
namespace kernel {
class SoftmaxGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SoftmaxGradGpuKernelMod() = default;
  ~SoftmaxGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_),
                                       kernel_name_ + " destroy output_descriptor failed");
  }

  void ResetResource() {
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() {
    output_size_list_.push_back(output_size_);
    if (use_workspace_) {
      workspace_size_list_.push_back(input_size_);
      workspace_size_list_.push_back(input_size_);
      workspace_size_list_.push_back(output_size_);
    }
    return;
  }

  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using SoftmaxGradGpuLaunchFunc =
    std::function<bool(SoftmaxGradGpuKernelMod *, const std::vector<KernelTensor *> &,
                       const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &, void *)>;

  void InitSizeByAxis(const std::vector<size_t> input_shape, const int axis) {
    axis_ = axis;
    if (axis_ < 0) {
      axis_ += SizeToInt(shape_size_);
    }
    if (axis_ >= SizeToInt(shape_size_) || axis_ < 0) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' must be in range [-" << shape_size_
                        << ", " << shape_size_ << "), but got " << axis;
    }

    input_shape_ = input_shape;
    transpose_shape_ = input_shape;
    for (size_t i = 0; i < input_shape.size(); ++i) {
      transpose_axis_.emplace_back(i);
    }
    std::swap(transpose_shape_[IntToSize(axis_)], transpose_shape_.back());
    std::swap(transpose_axis_[IntToSize(axis_)], transpose_axis_.back());

    size_t size_ = std::accumulate(input_shape.begin(), input_shape.end(), 1UL, std::multiplies<size_t>());
    channel_size_ = transpose_shape_.back();
    if (channel_size_ != 0) {
      batch_size_ = size_ / channel_size_;
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the value of the shape of the input along the 'axis' dimension should be greater than 0"
                        << ", but got " << channel_size_;
    }
    height_ = 1;
    width_ = 1;
    input_size_ = type_id_size_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
  }

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnTensorDescriptor_t y_desc_{nullptr};
  cudnnSoftmaxAlgorithm_t algo_{CUDNN_SOFTMAX_ACCURATE};
  cudnnSoftmaxMode_t mode_{CUDNN_SOFTMAX_MODE_INSTANCE};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  bool is_null_input_{false};
  bool use_workspace_{false};
  size_t input_size_{0};
  size_t output_size_{0};
  std::vector<size_t> input_shape_;
  std::vector<size_t> transpose_shape_;
  std::vector<size_t> transpose_axis_;
  int axis_{0};
  size_t shape_size_{0};
  size_t batch_size_{0};
  size_t channel_size_{0};
  size_t height_{0};
  size_t width_{0};
  size_t type_id_size_{0};
  SoftmaxGradGpuLaunchFunc kernel_func_;

 private:
  static std::vector<std::pair<KernelAttr, SoftmaxGradGpuLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
