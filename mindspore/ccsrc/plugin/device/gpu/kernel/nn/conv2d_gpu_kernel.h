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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2DGPUKERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2DGPUKERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
#define NBDIMS 4
template <typename T>
class Conv2dFwdGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  Conv2dFwdGpuKernelMod() { ResetResource(); }
  ~Conv2dFwdGpuKernelMod() override { DestroyResource(); }

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
      if (data_format_ == kOpFormat_NHWC) {
        CalPadNHWC(padded_size_ / sizeof(T), input_addr, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                   old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr,
                   reinterpret_cast<cudaStream_t>(stream_ptr));
      } else {
        CalPad(padded_size_ / sizeof(T), input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
               old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr,
               reinterpret_cast<cudaStream_t>(stream_ptr));
      }
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
  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = AnfAlgo::GetInputFormat(kernel_node, 0);
    auto format_attr = GetAttr<std::string>(kernel_node, "format");
    if (format_attr == kOpFormat_NHWC) {
      data_format_ = kOpFormat_NHWC;
    }
    auto in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    auto output_shape = AnfAlgo::GetOutputDeviceShape(kernel_node, 0);

    if (AnfAlgo::IsShapesDynamic({in_shape, filter_shape, output_shape})) {
      return true;
    }

    is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "x") ||
                     CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") ||
                     CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
    if (is_null_input_) {
      InitSizeLists();
      return true;
    }
    CheckTensorSize({in_shape, filter_shape, output_shape});
    SetNCHW(in_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
    if (data_format_ == kOpFormat_NHWC) {
      compute_format_ = CUDNN_TENSOR_NHWC;
    }
    Set4DDesc(in_shape, filter_shape, output_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (pad_list.size() != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got " << pad_list.size();
    }

    pad_height_ = pad_list[0];
    pad_width_ = pad_list[2];
    use_pad_ = !((pad_height_ == pad_list[1]) && (pad_width_ == pad_list[3]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
    if (pad_height_ == -1 || pad_width_ == -1) {
      return true;
    }

    SetStrideAndDilation(kernel_node);
    cudnnTensorDescriptor_t input_descriptor_real = nullptr;
    int padA[2];
    int strideA[2] = {stride_[2], stride_[3]};
    int dilaA[2] = {dilation_[2], dilation_[3]};
    if (use_pad_) {
      pad_height_ = pad_list[0] + pad_list[1];
      pad_width_ = pad_list[2] + pad_list[3];
      pad_top_ = pad_list[0];
      pad_left_ = pad_list[2];
      const int nbDims = 4;
      int dimA[NBDIMS];
      int strideApadded[NBDIMS];
      if (data_format_ == kOpFormat_NCHW || data_format_ == kOpFormat_DEFAULT) {
        ShapeVector padded_shape = {n_, c_, old_height_ + pad_height_, old_width_ + pad_width_};
        SetDimA(padded_shape, dimA, nbDims, data_format_);
        SetStrideA(padded_shape, strideApadded, nbDims, data_format_);
      } else if (data_format_ == kOpFormat_NHWC) {
        ShapeVector padded_shape = {n_, old_height_ + pad_height_, old_width_ + pad_width_, c_};
        SetDimA(padded_shape, dimA, nbDims, data_format_);
        SetStrideA(padded_shape, strideApadded, nbDims, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, 4, dimA, strideApadded),
                                  "cudnnSetTensor4dDescriptor failed");
      padA[0] = 0;
      padA[1] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolutionNdDescriptor failed");
      input_descriptor_real = padded_desc_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[0] = pad_height_;
      padA[1] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnSetConvolutionNdDescriptor(conv_desc_, 2, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      input_descriptor_real = input_desc_;
    }
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
    old_height_ = 0;
    old_width_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
    n_ = 0;
    c_ = 0;
    stride_.clear();
    dilation_.clear();
    group_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Conv2d";
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
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    if (input_num != 2) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (output_num != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    }
  }

  void Set4DDesc(const ShapeVector &in_shape, const ShapeVector &filter_shape, const ShapeVector &output_shape) {
    const int nbDims = 4;
    int dimA[NBDIMS];
    int strideAin[NBDIMS];
    int dimAout[NBDIMS];
    int strideAout[NBDIMS];
    SetDimA(in_shape, dimA, nbDims, data_format_);
    SetStrideA(in_shape, strideAin, nbDims, data_format_);
    SetDimA(output_shape, dimAout, nbDims, data_format_);
    SetStrideA(output_shape, strideAout, nbDims, data_format_);
    int filterDimA[4];
    // OHWI for NHWC; OIHW for NCHW
    SetDimA(filter_shape, filterDimA, nbDims, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, nbDims, dimA, strideAin),
                                "cudnnSetTensor4dDescriptor failed");

    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, nbDims, filterDimA),
      "cudnnSetFilter4dDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, nbDims, dimAout, strideAout),
                                "cudnnSetTensor4dDescriptor failed");
  }
  void SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
    constexpr int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionFwdAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_,
                                             output_desc_, requested_algo_count, &returned_algo_count, &perf_results),
      "cudnnGetConvolutionForwardAlgorithm_v7 failed");
    conv_algorithm_ = perf_results.algo;
#if CUDNN_VERSION < 8000
    if (group_ > 1) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnGetConvolutionForwardAlgorithm(
                                    cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                    CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm_),
                                  "cudnnGetConvolutionForwardAlgorithm failed");
    }
#endif
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    }
  }
  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "stride");
    std::vector<int64_t> dilation_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilation");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 4, but got " << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != 4) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
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
  std::string data_format_ = kOpFormat_NCHW;

  const float pad_value_ = 0.0;
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
  std::vector<int> stride_;
  std::vector<int> dilation_;
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

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV2DGPUKERNEL_H_
