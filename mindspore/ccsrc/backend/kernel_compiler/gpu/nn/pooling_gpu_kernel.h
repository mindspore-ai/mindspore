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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_

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
class PoolingGpuFwdKernel : public GpuKernel {
 public:
  PoolingGpuFwdKernel()
      : cudnn_handle_(nullptr),
        input_descriptor_(nullptr),
        output_descriptor_(nullptr),
        pooling_descriptor_(nullptr),
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
  ~PoolingGpuFwdKernel() override { DestroyResource(); }

  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }
  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = reinterpret_cast<T *>(inputs[0]->addr);
    T *output_addr = reinterpret_cast<T *>(outputs[0]->addr);
    const float alpha = 1;
    const float beta = 0;

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnPoolingForward(cudnn_handle_, pooling_descriptor_, &alpha, input_descriptor_,
                                                    input_addr, &beta, output_descriptor_, output_addr),
                                "cudnnPoolingForward failed");
    return true;
  }
  bool Init(const CNodePtr &kernel_node) {
    kernel_node_ = kernel_node;
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    auto input_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);

    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(input_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "PoolingGpuFwdKernel input is null.";
      InitSizeLists();
      return true;
    }
    SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    const int nbDims = 4;
    int dimA[4];
    int strideAin[4];
    int dimAout[4];
    int strideAout[4];
    SetDimA(input_shape, dimA, 4, data_format_);
    SetStrideA(input_shape, strideAin, 4, data_format_);
    SetDimA(output_shape, dimAout, 4, data_format_);
    SetStrideA(output_shape, strideAout, 4, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(input_descriptor_, cudnn_data_type_, nbDims, dimA, strideAin),
      "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetTensorNdDescriptor(output_descriptor_, cudnn_data_type_, nbDims, dimAout, strideAout),
      "cudnnSetTensor4dDescriptor failed");
    SetPoolingMode(kernel_node);
    SetPad(kernel_node);
    InitSizeLists();
    return true;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyPoolingDescriptor(pooling_descriptor_),
                               "cudnnDestroyPoolingDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreatePoolingDescriptor(&pooling_descriptor_),
                                "cudnnCreatePoolingDescriptor failed");
  }
  void InitSizeLists() {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(input_descriptor_, reinterpret_cast<size_t *>(&input_size_)),
        "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnGetTensorSizeInBytes(output_descriptor_, reinterpret_cast<size_t *>(&output_size_)),
        "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    output_size_list_.push_back(output_size_);
    return;
  }

 private:
  bool CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 1) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but pooling needs 1 inputs.";
      return false;
    }
    return true;
  }

  void SetPoolingMode(const CNodePtr &kernel_node) {
    mode_ = AnfAlgo::GetCNodeName(kernel_node);
    if (mode_ == "AvgPool") {
      pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
      pad_value_ = 0.0;
    } else {
      pooling_mode_ = CUDNN_POOLING_MAX;
      pad_value_ = kSignedMinFloat;
    }
  }
  void SetPad(const CNodePtr &kernel_node) {
    pad_mode_ = GetValue<std::string>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("pad_mode"));
    std::vector<int> window;
    std::vector<int64_t> window_me =
      GetValue<std::vector<int64_t>>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("kernel_size"));
    (void)std::transform(window_me.begin(), window_me.end(), std::back_inserter(window),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int window_height = window[2];
    int window_width = window[3];
    std::vector<int64_t> stride_me =
      GetValue<std::vector<int64_t>>(AnfAlgo::GetCNodePrimitive(kernel_node)->GetAttr("strides"));
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    int windowDimA[2] = {window_height, window_width};
    int paddingA[2] = {0, 0};
    int strideA[2] = {stride_[2], stride_[3]};
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
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
      pad_height_ = 0;
      pad_width_ = 0;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN,
                                                            2, windowDimA, paddingA, strideA),
                                "cudnnSetPoolingNdDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_descriptor_;
  cudnnTensorDescriptor_t output_descriptor_;
  cudnnPoolingDescriptor_t pooling_descriptor_;
  cudnnPoolingMode_t pooling_mode_ = CUDNN_POOLING_MAX;
  std::vector<int> stride_;
  std::string mode_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCHW;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
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

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_POOLING_GPU_KERNEL_H_
