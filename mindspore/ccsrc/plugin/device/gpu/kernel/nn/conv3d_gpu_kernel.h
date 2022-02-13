/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/pad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr size_t kInputDimSize = 5;
constexpr size_t kInDimIdxForN = 0;
constexpr size_t kInDimIdxForC = 1;
constexpr size_t kInDimIdxForD = 2;
constexpr size_t kInDimIdxForH = 3;
constexpr size_t kInDimIdxForW = 4;

constexpr size_t k3DPadSize = 6;
constexpr size_t kHead3DPadIdx = 0;
constexpr size_t kTail3DPadIdx = 1;
constexpr size_t kTop3DPadIdx = 2;
constexpr size_t kBottom3DPadIdx = 3;
constexpr size_t kLeft3DPadIdx = 4;
constexpr size_t kRight3DPadIdx = 5;

constexpr size_t kPadDepthIdx = 0;
constexpr size_t kPadHeightIdx = 1;
constexpr size_t kPadWidthIdx = 2;

constexpr size_t k3DStrideSize = 5;
constexpr size_t kDepth3DStrideIdx = 2;
constexpr size_t kHeight3DStrideIdx = 3;
constexpr size_t kWidth3DStrideIdx = 4;

constexpr size_t k3DDilationSize = 5;
constexpr size_t kDepth3DDilationIdx = 2;
constexpr size_t kHeight3DDilationIdx = 3;
constexpr size_t kWidth3DDilationIdx = 4;

template <typename T>
class Conv3dGpuKernelMod : public NativeGpuKernelMod {
 public:
  Conv3dGpuKernelMod() { ResetResource(); }
  ~Conv3dGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *input_addr = GetDeviceAddress<T>(inputs, 0);
    T *filter_addr = GetDeviceAddress<T>(inputs, 1);
    T *output_addr = GetDeviceAddress<T>(outputs, 0);
    T *workspace_addr = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    const float beta = 0;
    if (use_pad_) {
      T *padded_addr = GetDeviceAddress<T>(workspace, 1);
      CalPad3d(padded_size_ / sizeof(T), input_addr, n_, c_, old_depth_, old_height_, old_width_,
               old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_, pad_head_, pad_top_,
               pad_left_, pad_value_, padded_addr, reinterpret_cast<cudaStream_t>(stream_ptr));
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_,
                                conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
        "cudnnConvolutionForward failed");
    }

    return true;
  }

  void CheckSize(const size_t value, const size_t expect_value, const string arg_name) {
    if (value != expect_value) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of " << arg_name << " should be "
                        << expect_value << ", but got " << value;
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = kOpFormat_NCDHW;
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);
    is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "x") ||
                     CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({in_shape});
    (void)CheckSize(in_shape.size(), kInputDimSize, "x");
    n_ = SizeToInt(in_shape[kInDimIdxForN]);
    c_ = SizeToInt(in_shape[kInDimIdxForC]);
    old_depth_ = SizeToInt(in_shape[kInDimIdxForD]);
    old_height_ = SizeToInt(in_shape[kInDimIdxForH]);
    old_width_ = SizeToInt(in_shape[kInDimIdxForW]);
    compute_format_ = CUDNN_TENSOR_NCHW;
    SetNDDesc(in_shape, filter_shape, output_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (pad_list.size() != k3DPadSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' should be 6, but got " << pad_list.size();
    }
    SetPad(kernel_node, pad_list);
    SetStrideAndDilation(kernel_node);
    auto input_descriptor_real = GetInputDescReal(pad_list);
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(input_descriptor_real);
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    input_desc_ = nullptr;
    output_desc_ = nullptr;
    filter_desc_ = nullptr;
    conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    conv_desc_ = nullptr;
    padded_desc_ = nullptr;
    cudnn_data_type_ = CUDNN_DATA_FLOAT;
    compute_format_ = CUDNN_TENSOR_NCHW;
    old_depth_ = 0;
    old_height_ = 0;
    old_width_ = 0;
    pad_depth_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_head_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    n_ = 0;
    c_ = 0;
    stride_.clear();
    dilation_.clear();
    group_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Conv3d";
    input_size_ = 0;
    filter_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(filter_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(output_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(input_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&input_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&output_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&filter_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)),
                                  "cudnnGetFilterSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
                                  "cudnnGetTensorSizeInBytes failed");
    }
    input_size_list_.push_back(input_size_);
    input_size_list_.push_back(filter_size_);
    output_size_list_.push_back(output_size_);
    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, output_desc_,
                                                conv_algorithm_, &workspace_size_),
        "cudnnGetConvolutionForwardWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(
          kernel_node_,
          cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                  conv_algorithm_, &workspace_size_),
          "cudnnGetConvolutionForwardWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);

    return;
  }

 private:
  void CheckParam(const CNodePtr &kernel_node) {
    size_t input_num = AnfAlgo::GetInputTensorNum(kernel_node);
    const size_t kInputNum = 2;
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs should be 2, but got " << input_num;
    }
    size_t output_num = AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << output_num;
    }
  }

  void SetNDDesc(const std::vector<size_t> &in_shape, const std::vector<size_t> &filter_shape,
                 const std::vector<size_t> &output_shape) {
    const int kDims = 5;
    int dimA[kDims];
    int strideAin[kDims];
    int dimAout[kDims];
    int strideAout[kDims];
    int filterDimA[kDims];
    SetDimA(in_shape, dimA, kDims, data_format_);
    SetStrideA(in_shape, strideAin, kDims, data_format_);
    SetDimA(output_shape, dimAout, kDims, data_format_);
    SetStrideA(output_shape, strideAout, kDims, data_format_);
    SetDimA(filter_shape, filterDimA, kDims, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, kDims, dimA, strideAin),
                                "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, kDims, filterDimA),
      "cudnnSetFilter4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, kDims, dimAout, strideAout),
                                "cudnnSetTensor4dDescriptor failed");
  }

  void SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_,
                                             output_desc_, requested_algo_count, &returned_algo_count, &perf_results),
      "cudnnGetConvolutionForwardAlgorithm_v7 failed");
    conv_algorithm_ = perf_results.algo;
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k3DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' should be 5, but got "
                        << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis should be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != k3DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' should be 5, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis should be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }

  void SetPad(const CNodePtr &kernel_node, const std::vector<int> &pad_list) {
    pad_depth_ = pad_list[kHead3DPadIdx];
    pad_height_ = pad_list[kTop3DPadIdx];
    pad_width_ = pad_list[kLeft3DPadIdx];
    use_pad_ = (pad_depth_ != pad_list[kTail3DPadIdx]) || (pad_height_ != pad_list[kBottom3DPadIdx]) ||
               (pad_width_ != pad_list[kRight3DPadIdx]);
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
  }

  cudnnTensorDescriptor_t GetInputDescReal(const std::vector<int> &pad_list) {
    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    const int kNumDims = 5;
    const int kConvDims = 3;
    int padA[kConvDims];
    int strideA[kConvDims] = {stride_[kDepth3DStrideIdx], stride_[kHeight3DStrideIdx], stride_[kWidth3DStrideIdx]};
    int dilaA[kConvDims] = {dilation_[kDepth3DDilationIdx], dilation_[kHeight3DDilationIdx],
                            dilation_[kWidth3DDilationIdx]};
    if (use_pad_) {
      pad_depth_ = pad_list[kHead3DPadIdx] + pad_list[kTail3DPadIdx];
      pad_height_ = pad_list[kTop3DPadIdx] + pad_list[kBottom3DPadIdx];
      pad_width_ = pad_list[kLeft3DPadIdx] + pad_list[kRight3DPadIdx];
      pad_head_ = pad_list[kHead3DPadIdx];
      pad_top_ = pad_list[kTop3DPadIdx];
      pad_left_ = pad_list[kLeft3DPadIdx];
      int dimA[kNumDims];
      int strideApadded[kNumDims];
      if (data_format_ != kOpFormat_NCDHW) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'data_format' only support 'NCDHW' right now "
                          << ", but got " << data_format_;
      }
      auto padded_shape = {IntToSize(n_), IntToSize(c_), IntToSize(old_depth_ + pad_depth_),
                           IntToSize(old_height_ + pad_height_), IntToSize(old_width_ + pad_width_)};
      SetDimA(padded_shape, dimA, kNumDims, data_format_);
      SetStrideA(padded_shape, strideApadded, kNumDims, data_format_);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, kNumDims, dimA, strideApadded),
        "cudnnSetTensor4dDescriptor failed");
      padA[kPadDepthIdx] = 0;
      padA[kPadHeightIdx] = 0;
      padA[kPadWidthIdx] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      input_descriptor_real = padded_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[kPadDepthIdx] = pad_depth_;
      padA[kPadHeightIdx] = pad_height_;
      padA[kPadWidthIdx] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolution2dDescriptor failed");
      input_descriptor_real = input_desc_;
    }
    return input_descriptor_real;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnTensorDescriptor_t input_desc_;
  cudnnTensorDescriptor_t output_desc_;
  cudnnFilterDescriptor_t filter_desc_;
  cudnnConvolutionFwdAlgo_t conv_algorithm_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t padded_desc_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCDHW;

  const float pad_value_ = 0.0;
  cudnnDataType_t cudnn_data_type_;
  cudnnTensorFormat_t compute_format_;
  int old_depth_;
  int old_height_;
  int old_width_;
  int pad_depth_;
  int pad_height_;
  int pad_width_;
  int pad_head_;
  int pad_top_;
  int pad_left_;
  int n_;
  int c_;
  std::vector<int> stride_;
  std::vector<int> dilation_;
  int group_;
  bool is_null_input_;
  std::string kernel_name_;
  size_t input_size_;
  size_t filter_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GPU_KERNEL_H_
