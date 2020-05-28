/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/pad_impl.cuh"
#include "kernel/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PoolingGradGpuFwdKernel : public GpuKernel {
 public:
  PoolingGradGpuFwdKernel()
      : cudnn_handle_(nullptr),
        pooling_descriptor_(nullptr),
        y_descriptor_(nullptr),
        dy_descriptor_(nullptr),
        x_descriptor_(nullptr),
        dx_descriptor_(nullptr),
        padded_descriptor_(nullptr),
        pooling_mode_(CUDNN_POOLING_MAX),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        pad_value_(0),
        is_null_input_(false),
        input_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~PoolingGradGpuFwdKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *x_data = GetDeviceAddress<T>(inputs, 0);
    T *y = GetDeviceAddress<T>(inputs, 1);
    T *dy = GetDeviceAddress<T>(inputs, 2);
    T *dx = GetDeviceAddress<T>(outputs, 0);

    const float alpha = 1;
    const float beta = 0;
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 0);
      T *padded_dx = GetDeviceAddress<T>(workspace, 1);

      CalPad(padded_size_ / sizeof(T), x_data, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
             old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
             reinterpret_cast<cudaStream_t>(stream_ptr));

      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy,
                             padded_descriptor_, padded, &beta, padded_descriptor_, padded_dx),
        "cudnnPoolingBackward failed");

      CalPadGrad(output_size_ / sizeof(T), padded_dx, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
                 old_width_ + pad_width_, pad_top_, pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy,
                             x_descriptor_, x_data, &beta, dx_descriptor_, dx),
        "cudnnPoolingBackward failed");
    }
    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto window = GetAttr<std::vector<int>>(kernel_node, "ksize");
    int window_height = window[2];
    int window_width = window[3];
    SetPoolingMode(kernel_node);
    auto input_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto input_mask = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(input_mask);
    if (is_null_input_) {
      MS_LOG(WARNING) << "PoolingGradGpuFwdKernel input is null.";
      InitSizeLists();
      return true;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(y_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(input_mask[0]),
                                 SizeToInt(input_mask[1]), SizeToInt(input_mask[2]), SizeToInt(input_mask[3])),
      "cudnnSetTensor4dDescriptor");

    auto dout_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 2);
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(dy_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(dout_shape[0]),
                                 SizeToInt(dout_shape[1]), SizeToInt(dout_shape[2]), SizeToInt(dout_shape[3])),
      "cudnnSetTensor4dDescriptor");

    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(dx_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(output_shape[0]),
                                 SizeToInt(output_shape[1]), SizeToInt(output_shape[2]), SizeToInt(output_shape[3])),
      "cudnnSetTensor4dDescriptor failed");
    if (kSamePadModeUpperCase == pad_mode_ || kSamePadModeLowerCase == pad_mode_) {
      SetPad(input_shape, window_height, window_width);
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetPooling2dDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, window_height,
                                    window_width, pad_height_, pad_width_, stride_[2], stride_[3]),
        "cudnnSetPooling2dDescriptor failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetTensor4dDescriptor(x_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(input_shape[0]),
                                   SizeToInt(input_shape[1]), SizeToInt(input_shape[2]), SizeToInt(input_shape[3])),
        "cudnnSetTensor4dDescriptor");
    }
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&y_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dy_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&x_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&dx_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&padded_descriptor_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreatePoolingDescriptor(&pooling_descriptor_),
                                "cudnnCreatePoolingDescriptor failed");
  }
  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(y_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dx_descriptor_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(dy_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);

    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(x_descriptor_, &input_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);

    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      if (padded_size_ == 0) {
        MS_LOG(EXCEPTION) << "Padded size is 0.";
      }
      workspace_size_list_.push_back(padded_size_);
      workspace_size_list_.push_back(padded_size_);
    }
    return;
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but PoolingGradGpuFwdKernel needs 3 inputs.";
      return false;
    }
    return true;
  }
  void SetPad(const std::vector<size_t> &input_shape, const int &window_height, const int &window_width) {
    n_ = SizeToInt(input_shape[0]);
    c_ = SizeToInt(input_shape[1]);
    old_height_ = SizeToInt(input_shape[2]);
    old_width_ = SizeToInt(input_shape[3]);
    pad_height_ =
      std::max<int>(0, (((old_height_ / stride_[2]) * stride_[2] == old_height_ ? (old_height_ / stride_[2])
                                                                                : (old_height_ / stride_[2]) + 1) -
                        1) *
                           stride_[2] +
                         window_height - old_height_);
    pad_width_ =
      std::max<int>(0, (((old_width_ / stride_[3]) * stride_[3] == old_width_ ? (old_width_ / stride_[3])
                                                                              : (old_width_ / stride_[3]) + 1) -
                        1) *
                           stride_[3] +
                         window_width - old_width_);
    pad_top_ = pad_height_ / 2;
    pad_left_ = pad_width_ / 2;
    if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
      use_pad_ = false;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(padded_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_,
                                                           c_, old_height_ + pad_height_, old_width_ + pad_width_),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(x_descriptor_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(input_shape[0]),
                                 SizeToInt(input_shape[1]), SizeToInt(input_shape[2]) + (use_pad_ ? pad_height_ : 0),
                                 SizeToInt(input_shape[3]) + (use_pad_ ? pad_width_ : 0)),
      "cudnnSetTensor4dDescriptor");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetPooling2dDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            window_height, window_width, use_pad_ ? 0 : pad_top_,
                                                            use_pad_ ? 0 : pad_left_, stride_[2], stride_[3]),
                                "cudnnSetPooling2dDescriptor failed");
  }
  void SetPoolingMode(const CNodePtr &kernel_node) {
    pad_mode_ = GetAttr<std::string>(kernel_node, "padding");
    stride_ = GetAttr<std::vector<int>>(kernel_node, "strides");
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    mode_ = AnfAlgo::GetCNodeName(kernel_node);
    if (mode_ == "AvgPoolGradGpu") {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyPoolingDescriptor(pooling_descriptor_),
                               "cudnnDestroyPoolingDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(padded_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dx_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(x_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(dy_descriptor_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(y_descriptor_), "cudnnDestroyTensorDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnTensorDescriptor_t y_descriptor_;
  cudnnTensorDescriptor_t dy_descriptor_;
  cudnnTensorDescriptor_t x_descriptor_;
  cudnnTensorDescriptor_t dx_descriptor_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::string mode_;
  std::string pad_mode_;
  cudnnDataType_t cudnn_data_type_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  float pad_value_;
  bool is_null_input_;
  size_t input_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
