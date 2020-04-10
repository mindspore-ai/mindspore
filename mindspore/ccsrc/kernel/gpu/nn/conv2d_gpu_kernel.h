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

#ifndef MINDSPORE_CCSRC_KERNEL_GPU_NN_CONV2DGPUKERNEL_H_
#define MINDSPORE_CCSRC_KERNEL_GPU_NN_CONV2DGPUKERNEL_H_

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
class Conv2dGpuFwdKernel : public GpuKernel {
 public:
  Conv2dGpuFwdKernel()
      : cudnn_handle_(nullptr),
        input_desc_(nullptr),
        output_desc_(nullptr),
        filter_desc_(nullptr),
        conv_desc_(nullptr),
        padded_desc_(nullptr),
        cudnn_data_type_(CUDNN_DATA_FLOAT),
        old_height_(0),
        old_width_(0),
        pad_height_(0),
        pad_width_(0),
        pad_top_(0),
        pad_left_(0),
        n_(0),
        c_(0),
        stride_(1),
        dilation_(0),
        group_(1),
        is_null_input_(false),
        input_size_(0),
        filter_size_(0),
        output_size_(0),
        padded_size_(0),
        workspace_size_(0),
        use_pad_(true) {}
  ~Conv2dGpuFwdKernel() override { DestroyResource(); }
  const std::vector<size_t> &GetInputSizeList() const override { return input_size_list_; }
  const std::vector<size_t> &GetOutputSizeList() const override { return output_size_list_; }
  const std::vector<size_t> &GetWorkspaceSizeList() const override { return workspace_size_list_; }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, uintptr_t stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *filter_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *workspace_addr = nullptr;
    if (workspace_size_ != 0) {
      workspace_addr = GetDeviceAddress<T>(workspace, 0);
    }

    const float alpha = 1;
    const float beta = 0;
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_) {
      T *padded_addr = GetDeviceAddress<T>(workspace, 1);
      CalPad(padded_size_ / sizeof(T), input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
             old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr,
             reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    }

    return true;
  }
  bool Init(const CNodePtr &kernel_node) override {
    InitResource();
    if (!CheckParam(kernel_node)) {
      return false;
    }
    auto in_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetPrevNodeOutputInferShape(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputInferShape(kernel_node, 0);
    is_null_input_ = CHECK_NULL_INPUT(in_shape);
    if (is_null_input_) {
      MS_LOG(WARNING) << "Conv2dGpuFwdKernel input is null.";
      InitSizeLists();
      return true;
    }
    Set4DDesc(in_shape, filter_shape, output_shape);
    group_ = GetAttr<int>(kernel_node, "group");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetConvolutionGroupCount(conv_desc_, group_), "cudnnSetConvGroupCount failed");
    pad_height_ = GetAttr<int>(kernel_node, "pad");
    pad_width_ = pad_height_;
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    auto stride_ori = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "stride");
    auto dilation_ori = AnfAlgo::GetNodeAttr<std::vector<int>>(kernel_node, "dilation");
    if (stride_ori.size() != 4 || stride_ori[2] != stride_ori[3]) {
      MS_LOG(EXCEPTION) << "conv2d only support equal stride, and stride must be 4d!";
    }
    if (stride_ori[0] != 1 || stride_ori[1] != 1) {
      MS_LOG(EXCEPTION) << "conv2d stride only support 1 in N axis and C axis!";
    }
    if (dilation_ori.size() != 4 || dilation_ori[2] != dilation_ori[3]) {
      MS_LOG(EXCEPTION) << "conv2d only support equal dilation, and dilation must be 4d!";
    }
    if (dilation_ori[0] != 1 || dilation_ori[1] != 1) {
      MS_LOG(EXCEPTION) << "conv2d dilation only support 1 in N axis and C axis!";
    }
    stride_ = stride_ori[2];
    dilation_ = dilation_ori[2];

    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {
      SetPad(in_shape, kernel_node);
      input_descriptor_real = use_pad_ ? padded_desc_ : input_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnSetConvolution2dDescriptor(conv_desc_, pad_height_, pad_width_, stride_, stride_, dilation_, dilation_,
                                        CUDNN_CROSS_CORRELATION, cudnn_data_type_),
        "cudnnSetConvolution2dDescriptor failed");
      input_descriptor_real = input_desc_;
    }
    SelectAlgorithm(input_descriptor_real);
    InitSizeLists();
    return true;
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&input_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&output_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateTensorDescriptor(&padded_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateFilterDescriptor(&filter_desc_), "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(filter_size_);
    output_size_list_.push_back(output_size_);
    if ((pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) && use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, output_desc_,
                                                conv_algorithm_, &workspace_size_),
        "cudnnGetConvolutionForwardWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                  conv_algorithm_, &workspace_size_),
          "cudnnGetConvolutionForwardWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);

    return;
  }

 private:
  void DestroyResource() noexcept {
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyFilterDescriptor(filter_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(padded_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(output_desc_), "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(cudnnDestroyTensorDescriptor(input_desc_), "cudnnDestroyTensorDescriptor failed");
  }
  bool CheckParam(const CNodePtr &kernel_node) {
    cudnn_data_type_ = kCudnnDtypeMap[TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0))];
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(ERROR) << "Input number is " << input_num << ", but conv2d needs 2 inputs.";
      return false;
    }

    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(ERROR) << "Output number is " << output_num << ", but conv2d needs 1 output.";
      return false;
    }
    return true;
  }
  void SetPad(const std::vector<size_t> &in_shape, const CNodePtr &kernel_node) {
    auto pad_list = GetAttr<std::vector<int>>(kernel_node, "pad_list");

    n_ = SizeToInt(in_shape[0]);
    c_ = SizeToInt(in_shape[1]);
    old_height_ = SizeToInt(in_shape[2]);
    old_width_ = SizeToInt(in_shape[3]);
    pad_height_ = pad_list[0] + pad_list[1];
    pad_width_ = pad_list[2] + pad_list[3];
    pad_top_ = pad_list[0];
    pad_left_ = pad_list[2];

    // if use_pad_ == true, using zero padding in advance, else using the default cudnn pad.
    if (pad_height_ % 2 == 0 && pad_width_ % 2 == 0) {
      use_pad_ = false;
    }
    CHECK_CUDNN_RET_WITH_EXCEPT(cudnnSetTensor4dDescriptor(padded_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, n_, c_,
                                                           old_height_ + pad_height_, old_width_ + pad_width_),
                                "cudnnSetTensor4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetConvolution2dDescriptor(conv_desc_, use_pad_ ? 0 : pad_top_, use_pad_ ? 0 : pad_left_, stride_, stride_,
                                      dilation_, dilation_, CUDNN_CROSS_CORRELATION, cudnn_data_type_),
      "cudnnSetConvolution2dDescriptor failed");
  }

  void Set4DDesc(const std::vector<size_t> &in_shape, const std::vector<size_t> &filter_shape,
                 const std::vector<size_t> &output_shape) {
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(input_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(in_shape[0]),
                                 SizeToInt(in_shape[1]), SizeToInt(in_shape[2]), SizeToInt(in_shape[3])),
      "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetFilter4dDescriptor(filter_desc_, cudnn_data_type_, CUDNN_TENSOR_NCHW, SizeToInt(filter_shape[0]),
                                 SizeToInt(filter_shape[1]), SizeToInt(filter_shape[2]), SizeToInt(filter_shape[3])),
      "cudnnSetFilter4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      cudnnSetTensor4dDescriptor(output_desc_, CUDNN_TENSOR_NCHW, cudnn_data_type_, SizeToInt(output_shape[0]),
                                 SizeToInt(output_shape[1]), SizeToInt(output_shape[2]), SizeToInt(output_shape[3])),
      "cudnnSetTensor4dDescriptor failed");
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
    if (group_ > 1 || CUDNN_MAJOR < 7) {
      CHECK_CUDNN_RET_WITH_EXCEPT(cudnnGetConvolutionForwardAlgorithm(
                                    cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm_),
                                  "cudnnGetConvolutionForwardAlgorithm failed");
    } else {
      constexpr int requested_algo_count = 1;
      int returned_algo_count;
      cudnnConvolutionFwdAlgoPerf_t perf_results;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_,
                                               output_desc_, requested_algo_count, &returned_algo_count, &perf_results),
        "cudnnGetConvolutionForwardAlgorithm_v7 failed");
      conv_algorithm_ = perf_results.algo;
    }
  }
  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_algorithm_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t padded_desc_;
  std::string pad_mode_;
  std::vector<size_t> input_size_list_;
  std::vector<size_t> output_size_list_;
  std::vector<size_t> workspace_size_list_;
  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  int old_height_;
  int old_width_;
  int pad_height_;
  int pad_width_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  int stride_;
  int dilation_;
  int group_;
  bool is_null_input_;
  size_t input_size_;
  size_t filter_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_KERNEL_GPU_NN_CONV2DGPUKERNEL_H_
