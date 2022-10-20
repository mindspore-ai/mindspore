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

#include <vector>
#include <string>
#include <algorithm>
#include <map>
#include <utility>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
namespace mindspore {
namespace kernel {
class SoftmaxGradGpuKernelMod : public NativeGpuKernelMod {
 public:
  SoftmaxGradGpuKernelMod() = default;
  ~SoftmaxGradGpuKernelMod() override { DestroyResource(); }

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

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(y_desc_),
                                       kernel_name_ + " destroy output_descriptor failed");
  }

  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);
  using SoftmaxGradGpuLaunchFunc =
    std::function<bool(SoftmaxGradGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;

 private:
  void InitSizeByAxis(const std::vector<size_t> input_shape, const int axis) {
    axis_ = axis;
    if (axis_ < 0) {
      axis_ += SizeToInt(shape_size_);
    }
    if (axis_ == 1) {
      batch_size_ = input_shape[0];
      channel_size_ = input_shape[1];
    } else if (axis_ == 0) {
      batch_size_ = input_shape[1];
      channel_size_ = input_shape[0];
      input_shape_.push_back(input_shape[0]);
      input_shape_.push_back(input_shape[1]);
      transpose_shape_.push_back(input_shape[1]);
      transpose_shape_.push_back(input_shape[0]);
      transpose_axis_.push_back(1);
      transpose_axis_.push_back(0);
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' must be in range [-" << shape_size_
                        << ", " << shape_size_ << "), but got " << axis;
    }

    height_ = 1;
    width_ = 1;
    input_size_ = type_id_size_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(size_t);
  }

  SoftmaxGradGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, SoftmaxGradGpuLaunchFunc>> func_list_;

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnTensorDescriptor_t y_desc_{nullptr};
  cudnnSoftmaxAlgorithm_t algo_{CUDNN_SOFTMAX_ACCURATE};
  cudnnSoftmaxMode_t mode_{CUDNN_SOFTMAX_MODE_INSTANCE};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  bool is_null_input_{false};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t workspace_size_{0};

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
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
