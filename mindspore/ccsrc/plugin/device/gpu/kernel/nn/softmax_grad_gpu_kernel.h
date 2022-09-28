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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SoftmaxGradGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  SoftmaxGradGpuKernelMod()
      : cudnn_handle_(nullptr),
        y_desc_(nullptr),
        algo_(CUDNN_SOFTMAX_ACCURATE),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_null_input_(false),
        kernel_name_("SoftmaxGrad"),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        axis_(0),
        shape_size_(0),
        batch_size_(0),
        channel_size_(0),
        height_(0),
        width_(0) {}
  ~SoftmaxGradGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *y_addr = GetDeviceAddress<T>(inputs, 0);
    T *dy_addr = GetDeviceAddress<T>(inputs, 1);
    T *dx_addr = GetDeviceAddress<T>(outputs, 0);

    T *transpose_y_addr = GetDeviceAddress<T>(workspace, 0);
    T *transpose_dy_addr = GetDeviceAddress<T>(workspace, 1);
    T *transpose_dx_addr = GetDeviceAddress<T>(workspace, 2);
    size_t *input_shape = GetDeviceAddress<size_t>(workspace, 3);
    size_t *transpose_shape = GetDeviceAddress<size_t>(workspace, 4);
    size_t *transpose_axis = GetDeviceAddress<size_t>(workspace, 5);
    const float alpha = 1;
    const float beta = 0;

    if (axis_ == 1) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, y_addr, y_desc_,
                                                       dy_addr, &beta, y_desc_, dx_addr),
                                  "cudnnSoftmaxBackward failed");
    } else {
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(transpose_shape, &transpose_shape_[0], workspace_size_,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(kernel_node_,
                                 cudaMemcpyAsync(transpose_axis, &transpose_axis_[0], workspace_size_,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_axis failed");
      size_t size = input_size_ / sizeof(T);
      CalTranspose(size, y_addr, input_shape, transpose_axis, shape_size_, transpose_y_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      CalTranspose(size, dy_addr, input_shape, transpose_axis, shape_size_, transpose_dy_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, transpose_y_addr,
                                                       y_desc_, transpose_dy_addr, &beta, y_desc_, transpose_dx_addr),
                                  "cudnnSoftmaxBackward failed");
      CalTranspose(size, transpose_dx_addr, transpose_shape, transpose_axis, shape_size_, dx_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
    auto temp_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    if (AnfAlgo::IsShapesDynamic({temp_shape})) {
      InitSizeLists();
      return true;
    }
    std::vector<size_t> input_shape(temp_shape.begin(), temp_shape.end());
    auto axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
    if (axis == -1 || axis == SizeToInt(input_shape.size())) {
      axis = 1;

      std::vector<size_t> reshape;
      size_t dim0 = 1;
      for (size_t i = 0; i < input_shape.size() - 1; i++) {
        dim0 *= input_shape[i];
      }
      reshape.push_back(dim0);
      reshape.push_back(input_shape[input_shape.size() - 1]);
      input_shape = reshape;
    }

    is_null_input_ = CHECK_SHAPE_NULL(input_shape, kernel_name_, "input");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }

    shape_size_ = input_shape.size();
    if (shape_size_ > 3) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_
                        << "', the dimension of input must be less than and equal to 3, but got " << shape_size_;
    }
    auto kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "LogSoftmaxGrad") {
      algo_ = CUDNN_SOFTMAX_LOG;
      auto axis = static_cast<int>(GetAttr<int64_t>(kernel_node, "axis"));
      InitSizeByAxis(input_shape, axis);
    } else {
      algo_ = CUDNN_SOFTMAX_ACCURATE;
      std::vector<int> axis;
      std::vector<int64_t> axis_me = GetAttr<std::vector<int64_t>>(kernel_node, "axis");
      (void)std::transform(axis_me.begin(), axis_me.end(), std::back_inserter(axis),
                           [](const int64_t &value) { return static_cast<int>(value); });
      if (axis.size() < 1) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'axis' cannot be equal to 0, but got "
                          << axis.size();
      }
      InitSizeByAxis(input_shape, axis[0]);
    }

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                                 SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
      "set input_descriptor failed");
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(y_desc_), "destroy output_descriptor failed");
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    y_desc_ = nullptr;
    algo_ = CUDNN_SOFTMAX_ACCURATE;
    mode_ = CUDNN_SOFTMAX_MODE_INSTANCE;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    is_null_input_ = false;
    input_size_ = 0;
    output_size_ = 0;
    workspace_size_ = 0;
    axis_ = 0;
    shape_size_ = 0;
    batch_size_ = 0;
    channel_size_ = 0;
    height_ = 0;
    width_ = 0;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&y_desc_), "create input_descriptor failed");
  }

  void InitSizeLists() override {
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
    input_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(size_t);
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  cudnnDataType_t cudnn_data_type_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;

  std::vector<size_t> input_shape_;
  std::vector<size_t> transpose_shape_;
  std::vector<size_t> transpose_axis_;
  int axis_;
  size_t shape_size_;

  size_t batch_size_;
  size_t channel_size_;
  size_t height_;
  size_t width_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
