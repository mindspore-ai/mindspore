/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_

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
class SoftmaxGpuKernelMod : public NativeGpuKernelMod {
 public:
  SoftmaxGpuKernelMod() = default;
  ~SoftmaxGpuKernelMod() override { DestroyResource(); }

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

  using SoftmaxGpuLaunchFunc =
    std::function<bool(SoftmaxGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                       const std::vector<kernel::AddressPtr> &, const std::vector<kernel::AddressPtr> &, void *)>;

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(output_descriptor_),
                                       kernel_name_ + " destroy output_descriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(input_descriptor_),
                                       kernel_name_ + " destroy input_descriptor failed");
  }

  void ResetResource() {
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
    input_shape_.clear();
    transpose_shape_.clear();
    transpose_axis_.clear();
  }

 protected:
  void InitSizeLists() {
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(input_size_);
    workspace_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    workspace_size_list_.push_back(workspace_size_);
    return;
  }

 private:
  void InitSizeByAxis(const std::vector<size_t> &input_shape, const int &axis) {
    if (input_shape.size() == 2) {
      InitSizeByAxis2D(input_shape, axis);
    } else {
      int axis_pos = axis;
      if (axis_pos < 0) {
        axis_pos += input_shape.size();
      }

      if (axis_pos == SizeToInt(input_shape.size() - 1)) {
        InitSizeByAxisLastDim(input_shape, axis_pos);
      } else {
        InitSizeByAxisND(input_shape, axis_pos);
      }
    }
  }

  void InitSizeByAxis2D(const std::vector<size_t> &input_shape, const int &axis) {
    int axis_pos = axis;
    if (axis_pos < 0) {
      axis_pos += SizeToInt(shape_size_);
    }
    if (axis_pos == 1) {
      batch_size_ = input_shape[0];
      channel_size_ = input_shape[1];
      need_transpose_ = false;
    } else if (axis_pos == 0) {
      batch_size_ = input_shape[1];
      channel_size_ = input_shape[0];
      input_shape_.push_back(input_shape[0]);
      input_shape_.push_back(input_shape[1]);
      transpose_shape_.push_back(input_shape[1]);
      transpose_shape_.push_back(input_shape[0]);
      transpose_axis_.push_back(1);
      transpose_axis_.push_back(0);
      need_transpose_ = true;
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

  void InitSizeByAxisLastDim(const std::vector<size_t> &input_shape, const int &axis) {
    int axis_pos = axis;
    if (axis_pos < 0) {
      axis_pos += input_shape.size();
    }
    // axis must be -1 with ND
    if (axis_pos != SizeToInt(input_shape.size() - 1)) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' must be equal to -1 or "
                        << (input_shape.size() - 1) << ", but got " << axis;
    }
    // squeeze to 2d, then invoke cudnn
    size_t n = 1;
    for (size_t i = 0; i < input_shape.size() - 1; i++) {
      n *= input_shape[i];
    }

    batch_size_ = n;
    channel_size_ = input_shape[axis_pos];
    height_ = 1;
    width_ = 1;
    input_size_ = type_id_size_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    input_shape_.push_back(batch_size_);
    input_shape_.push_back(channel_size_);
    need_transpose_ = false;
  }

  void InitSizeByAxisND(const std::vector<size_t> &input_shape, const int &axis) {
    int axis_pos = axis;
    if (axis_pos < 0) {
      axis_pos += input_shape.size();
    }

    if (axis_pos >= SizeToInt(input_shape.size())) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'axis' must be in range [-" << input_shape.size()
                        << ", " << input_shape.size() << "), but got " << axis;
    }
    // n keep tracks of squeezed size
    size_t n = 1;
    for (int i = 0; i < SizeToInt(input_shape.size()); i++) {
      input_shape_.push_back(input_shape[i]);
      if (i == axis_pos) {
        size_t lastIndex = input_shape.size() - 1;
        transpose_shape_.push_back(input_shape[lastIndex]);
        transpose_axis_.push_back(lastIndex);
      } else if (i == SizeToInt(input_shape.size() - 1)) {
        transpose_shape_.push_back(input_shape[axis_pos]);
        transpose_axis_.push_back(axis_pos);
        n *= input_shape[i];
      } else {
        transpose_shape_.push_back(input_shape[i]);
        transpose_axis_.push_back(i);
        n *= input_shape[i];
      }
    }

    batch_size_ = n;
    channel_size_ = input_shape[axis_pos];
    height_ = 1;
    width_ = 1;
    input_size_ = type_id_size_ * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(size_t);
    need_transpose_ = true;
  }

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnTensorDescriptor_t input_descriptor_{nullptr};
  cudnnTensorDescriptor_t output_descriptor_{nullptr};
  cudnnSoftmaxAlgorithm_t algo_{CUDNN_SOFTMAX_ACCURATE};
  cudnnSoftmaxMode_t mode_{CUDNN_SOFTMAX_MODE_INSTANCE};
  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  bool is_null_input_{false};
  size_t input_size_{0};
  size_t output_size_{0};
  size_t workspace_size_{0};

  std::vector<size_t> input_shape_{};
  std::vector<size_t> transpose_shape_{};
  std::vector<size_t> transpose_axis_{};
  bool need_transpose_{false};
  size_t shape_size_{0};

  size_t batch_size_{0};
  size_t channel_size_{0};
  size_t height_{0};
  size_t width_{0};
  size_t type_id_size_{0};

  SoftmaxGpuLaunchFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, SoftmaxGpuLaunchFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_
