/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_

#include <vector>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/kernel_constants.h"
#include "kernel/gpu/cuda_impl/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SoftmaxGradGpuKernel : public GpuKernel {
 public:
  SoftmaxGradGpuKernel()
      : cudnn_handle_(nullptr),
        y_desc_(nullptr),
        algo_(CUDNN_SOFTMAX_ACCURATE),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        axis_(0),
        shape_size_(0),
        batch_size_(0),
        channel_size_(0),
        height_(0),
        width_(0) {}
  ~SoftmaxGradGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

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
    int *input_shape = GetDeviceAddress<int>(workspace, 3);
    int *transpose_shape = GetDeviceAddress<int>(workspace, 4);
    int *transpose_axis = GetDeviceAddress<int>(workspace, 5);
    const float alpha = 1;
    const float beta = 0;

    if (axis_ == 1) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, y_addr, y_desc_,
                                                       dy_addr, &beta, y_desc_, dx_addr),
                                  "cudnnSoftmaxBackward failed");
    } else {
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(input_shape, &input_shape_[0], workspace_size_, cudaMemcpyHostToDevice,
                                                 reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(transpose_shape, &transpose_shape_[0], workspace_size_,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_shape failed");
      CHECK_CUDA_RET_WITH_EXCEPT(cudaMemcpyAsync(transpose_axis, &transpose_axis_[0], workspace_size_,
                                                 cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_ptr)),
                                 "cudaMemcpyAsync input_axis failed");
      int size = SizeToInt(input_size_ / sizeof(T));
      CalTranspose(size, y_addr, input_shape, transpose_axis, shape_size_, transpose_y_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      CalTranspose(size, dy_addr, input_shape, transpose_axis, shape_size_, transpose_dy_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSoftmaxBackward(cudnn_handle_, algo_, mode_, &alpha, y_desc_, transpose_y_addr,
                                                       y_desc_, transpose_dy_addr, &beta, y_desc_, transpose_dx_addr),
                                  "cudnnSoftmaxBackward failed");
      CalTranspose(size, transpose_dx_addr, transpose_shape, transpose_axis, shape_size_, dx_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but softmax grad needs 2 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but softmax grad needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "SoftmaxGradGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    shape_size_ = SizeToInt(input_shape.size());
    if (shape_size_ != 2) {
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but softmax grad only supports 2-D inputs.";
    }
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "LogSoftmaxGrad") {
      algo_ = CUDNN_SOFTMAX_LOG;
      auto axis = GetAttr<int>(kernel_node, "axis");
      InitSizeByAxis(input_shape, axis);
    } else {
      algo_ = CUDNN_SOFTMAX_ACCURATE;
      auto axis = GetAttr<std::vector<int>>(kernel_node, "axis");
      InitSizeByAxis(input_shape, axis[0]);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(y_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                                 SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
      "set input_descriptor failed");
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&y_desc_), "create input_descriptor failed");
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
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(y_desc_), "destroy output_descriptor failed");
  }

  void InitSizeByAxis(const std::vector<size_t> input_shape, const int axis) {
    axis_ = axis;
    if (axis_ < 0) {
      axis_ += shape_size_;
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
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but axis(" << axis << ") is invalid.";
    }

    height_ = 1;
    width_ = 1;
    input_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = IntToSize(shape_size_) * sizeof(int);
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t y_desc_;
  cudnnSoftmaxAlgorithm_t algo_;
  cudnnSoftmaxMode_t mode_;
  cudnnDataType_t cudnn_data_type_;
  bool is_null_input_;
  size_t input_size_;
  size_t output_size_;
  size_t workspace_size_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;

  std::vector<int> input_shape_;
  std::vector<int> transpose_shape_;
  std::vector<int> transpose_axis_;
  int axis_;
  int shape_size_;

  size_t batch_size_;
  size_t channel_size_;
  size_t height_;
  size_t width_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_SOFTMAX_GRAD_GPU_KERNEL_H_
