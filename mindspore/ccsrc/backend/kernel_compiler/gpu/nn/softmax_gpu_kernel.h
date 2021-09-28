/**
 * Copyright 2019-2021 Huawei Technologies Co., Ltd
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
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/kernel_constants.h"
#include "backend/kernel_compiler/gpu/cuda_impl/transpose_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class SoftmaxGpuKernel : public GpuKernel {
 public:
  SoftmaxGpuKernel()
      : cudnn_handle_(nullptr),
        input_descriptor_(nullptr),
        output_descriptor_(nullptr),
        algo_(CUDNN_SOFTMAX_ACCURATE),
        mode_(CUDNN_SOFTMAX_MODE_INSTANCE),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        workspace_size_(0),
        need_transpose_(false),
        shape_size_(0),
        batch_size_(0),
        channel_size_(0),
        height_(0),
        width_(0) {}
  ~SoftmaxGpuKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    const float alpha = 1;
    const float beta = 0;

    if (need_transpose_ == false) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_,
                                                      input_addr, &beta, output_descriptor_, output_addr),
                                  "cudnnSoftmaxForward failed");
    } else {
      T *transpose_input_addr = GetDeviceAddress<T>(workspace, 0);
      T *transpose_output_addr = GetDeviceAddress<T>(workspace, 1);
      size_t *input_shape = GetDeviceAddress<size_t>(workspace, 2);
      size_t *transpose_shape = GetDeviceAddress<size_t>(workspace, 3);
      size_t *transpose_axis = GetDeviceAddress<size_t>(workspace, 4);
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
      CalTranspose(size, input_addr, input_shape, transpose_axis, shape_size_, transpose_input_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSoftmaxForward(cudnn_handle_, algo_, mode_, &alpha, input_descriptor_, transpose_input_addr, &beta,
                            output_descriptor_, transpose_output_addr),
        "cudnnSoftmaxForward failed");
      CalTranspose(size, transpose_output_addr, transpose_shape, transpose_axis, shape_size_, output_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
    }
    return true;
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_node_ = kernel_node;
    InitResource();
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but softmax needs 1 input.";
      return false;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but softmax needs 1 output.";
      return false;
    }
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "SoftmaxGpuKernel input is null";
      InitSizeLists();
      return true;
    }
    shape_size_ = input_shape.size();
    auto kernel_name = AnfAlgo::GetCNodeName(kernel_node);
    if (kernel_name == "LogSoftmax") {
      algo_ = CUDNN_SOFTMAX_LOG;
      auto axis = LongToInt(GetAttr<int64_t>(kernel_node, "axis"));
      InitSizeByAxis(input_shape, axis);
    } else {
      algo_ = CUDNN_SOFTMAX_ACCURATE;
      std::vector<int> axis;
      std::vector<int64_t> axis_me = GetAttr<std::vector<int64_t>>(kernel_node, "axis");
      (void)std::transform(axis_me.begin(), axis_me.end(), std::back_inserter(axis),
                           [](const int64_t &value) { return LongToInt(value); });
      if (axis.size() < 1) {
        MS_LOG(EXCEPTION) << "For 'SoftmaxGpuKernel', the rank of axis should be greater than or equal to 1, "
                          << "but got the rank of axis: " << axis.size();
      }
      InitSizeByAxis(input_shape, axis[0]);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(input_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                                 SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
      "set input_descriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnSetTensor4dDescriptor(output_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(batch_size_),
                                 SizeToInt(channel_size_), SizeToInt(height_), SizeToInt(width_)),
      "set output_descriptor failed");
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_descriptor_),
                               "destroy output_descriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_descriptor_),
                               "destroy input_descriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_descriptor_),
                                "create input_descriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_descriptor_),
                                "create output_descriptor failed");
  }

  void InitSizeLists() override {
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
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but axis(" << axis << ") is invalid.";
    }

    height_ = 1;
    width_ = 1;
    input_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(size_t);
  }

  void InitSizeByAxisLastDim(const std::vector<size_t> &input_shape, const int &axis) {
    int axis_pos = axis;
    if (axis_pos < 0) {
      axis_pos += input_shape.size();
    }
    // axis should be -1 with ND
    if (axis_pos != SizeToInt(input_shape.size() - 1)) {
      MS_LOG(EXCEPTION) << "Input is " << shape_size_ << "-D, but axis(" << axis << ") is invalid.";
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
    input_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    input_shape_.push_back(batch_size_);
    input_shape_.push_back(channel_size_);
    need_transpose_ = false;
  }

  void InitSizeByAxisND(const std::vector<size_t> &input_shape, const int &axis) {
    size_t axis_pos = axis;
    if (axis_pos < 0) {
      axis_pos += input_shape.size();
    }

    if (axis_pos >= input_shape.size()) {
      MS_LOG(EXCEPTION) << "For 'SoftmaxGpuKernel', the axis_pos should be less than the rank of input_shape, "
                        << "but got axis_pos: " << axis_pos << ", the rank of input_shape: " << input_shape.size();
    }
    // n keep tracks of squeezed size
    size_t n = 1;
    for (size_t i = 0; i < input_shape.size(); i++) {
      input_shape_.push_back(input_shape[i]);
      if (i == axis_pos) {
        size_t lastIndex = input_shape.size() - 1;
        transpose_shape_.push_back(input_shape[lastIndex]);
        transpose_axis_.push_back(lastIndex);
      } else if (i == (input_shape.size() - 1)) {
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
    input_size_ = sizeof(T) * batch_size_ * channel_size_ * height_ * width_;
    output_size_ = input_size_;
    workspace_size_ = shape_size_ * sizeof(size_t);
    need_transpose_ = true;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;
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

  std::vector<size_t> input_shape_;
  std::vector<size_t> transpose_shape_;
  std::vector<size_t> transpose_axis_;
  bool need_transpose_;
  size_t shape_size_;

  size_t batch_size_;
  size_t channel_size_;
  size_t height_;
  size_t width_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_SOFTMAX_GPU_KERNEL_H_
