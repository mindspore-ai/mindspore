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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <algorithm>
#include "backend/kernel_compiler/gpu/gpu_kernel.h"
#include "backend/kernel_compiler/gpu/gpu_kernel_factory.h"
#include "backend/kernel_compiler/gpu/cuda_impl/pad_impl.cuh"
#include "backend/kernel_compiler/gpu/kernel_constants.h"

namespace mindspore {
namespace kernel {
template <typename T>
class PoolingGradGpuKernel : public GpuKernel {
 public:
  PoolingGradGpuKernel()
      : cudnn_handle_(nullptr),
        pooling_descriptor_(nullptr),
        y_descriptor_(nullptr),
        dy_descriptor_(nullptr),
        x_descriptor_(nullptr),
        dx_descriptor_(nullptr),
        pooling_mode_(CUDNN_POOLING_MAX),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        compute_format_(CUDNN_TENSOR_NCHW),
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
        workspace_size_(0) {}
  ~PoolingGradGpuKernel() override { DestroyResource(); }

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
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy,
                           x_descriptor_, x_data, &beta, dx_descriptor_, dx),
      "cudnnPoolingBackward failed");
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
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto input_mask = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto dout_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 2);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape) || CHECK_NULL_INPUT(input_mask);
    if (is_null_input_) {
      MS_LOG(WARNING) << "PoolingGradGpuKernel input is null.";
      InitSizeLists();
      return true;
    }
    SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    int windowDimA[2] = {window_height, window_width};
    int paddingA[2] = {0, 0};
    int strideA[2] = {stride_[2], stride_[3]};
    const int nbDims = 4;
    int dimA[4];
    int strideAin[4];
    int dimAy[4];
    int strideAiny[4];
    int dimAdy[4];
    int strideAdy[4];
    int dimAout[4];
    int strideAout[4];
    SetDimA(input_shape, dimA, 4, data_format_);
    SetStrideA(input_shape, strideAin, 4, data_format_);
    SetDimA(input_mask, dimAy, 4, data_format_);
    SetStrideA(input_mask, strideAiny, 4, data_format_);
    SetDimA(dout_shape, dimAdy, 4, data_format_);
    SetStrideA(dout_shape, strideAdy, 4, data_format_);
    SetDimA(output_shape, dimAout, 4, data_format_);
    SetStrideA(output_shape, strideAout, 4, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(y_descriptor_, cudnn_data_type_, nbDims, dimAy, strideAiny),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(dy_descriptor_, cudnn_data_type_, nbDims, dimAdy, strideAdy),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensorNdDescriptor(dx_descriptor_, cudnn_data_type_, nbDims, dimAout, strideAout),
      "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensorNdDescriptor(x_descriptor_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensor4dDescriptor failed");
    if (kSamePadModeUpperCase == pad_mode_ || kSamePadModeLowerCase == pad_mode_) {
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
      paddingA[0] = pad_top_;
      paddingA[1] = pad_left_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            2, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
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
    return;
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 3) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but PoolingGradGpuKernel needs 3 inputs.";
      return false;
    }
    return true;
  }
  void SetPad(const std::vector<size_t> &input_shape, const int &window_height, const int &window_width) {
    n_ = SizeToInt(input_shape[0]);
    c_ = SizeToInt(input_shape[1]);
    old_height_ = SizeToInt(input_shape[2]);
    old_width_ = SizeToInt(input_shape[3]);
  }
  void SetPoolingMode(const CNodePtr &kernel_node) {
    pad_mode_ = GetAttr<std::string>(kernel_node, "padding");
    stride_ = GetAttr<std::vector<int>>(kernel_node, "strides");
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
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
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  std::string mode_;
  std::string pad_mode_;
  std::string data_format_ = "NCHW";
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
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
  size_t workspace_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GRAD_GPU_KERNEL_H_
