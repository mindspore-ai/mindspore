/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_CUDNN_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_CUDNN_GPU_KERNEL_H_

#include <cuda.h>
#include <cudnn.h>
#include <unordered_map>
#include <string>
#include <algorithm>
#include <vector>
#include <memory>
#include "kernel/gpu/cudnn/convolution/abstract_conv_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/convolution/convolution_ops_impl.cuh"
#include "kernel/gpu/cuda_impl/cuda_ops/pad_impl.cuh"

namespace mindspore {
namespace kernel {
template <typename T>
class ConvolutionCudnnGpuKernel : public AbstractConvolutionGpuKernel {
 public:
  explicit ConvolutionCudnnGpuKernel(enum ConvType conv_type) : AbstractConvolutionGpuKernel(conv_type) {}
  ~ConvolutionCudnnGpuKernel() {}

  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&tensor0_desc_),
                                        "Create tensor descriptor 0 failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&tensor1_desc_),
                                        "Create tensor descriptor 1 failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateFilterDescriptor(&filter_desc_), "Create filter descriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&padded_desc_), "Create padded descriptor failed.");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateConvolutionDescriptor(&conv_desc_),
                                        "Create convolution descriptor failed.");
  };
  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyConvolutionDescriptor(conv_desc_),
                                       "Destroy convolution descriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyFilterDescriptor(filter_desc_), "Destroy filter descriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(padded_desc_), "Destroy padded descriptor failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(tensor0_desc_),
                                       "Destroy tensor descriptor 0 failed.");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(cudnnDestroyTensorDescriptor(tensor1_desc_),
                                       "Destroy tensor descriptor 1 failed.");
  }
  void ResetResource(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                     std::vector<size_t> *output_size_list,
                     std::vector<size_t> *workspace_size_list) noexcept override {
    forward_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    input_grad_algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;
    filter_grad_algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;

    conv_args->in_height = 0;
    conv_args->in_width = 0;
    conv_args->pad_height = 0;
    conv_args->pad_width = 0;
    conv_args->pad_top = 0;
    conv_args->pad_left = 0;
    conv_args->use_pad = false;
    conv_args->batch_size = 0;
    conv_args->in_channel = 0;
    tensor0_size_ = 0;
    tensor1_size_ = 0;
    filter_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    input_size_list->clear();
    output_size_list->clear();
    workspace_size_list->clear();
  }

 private:
  void CalConv2dPadNCHW(const ConvolutionArgs &conv_args, const void *input_addr, void *padded_addr, void *stream_ptr);

  void CalConv2dPadNHWC(const ConvolutionArgs &conv_args, const void *input_addr, void *padded_addr, void *stream_ptr);

  void CalConv2dPadGradNCHW(const ConvolutionArgs &conv_args, const void *padded_addr, void *dx_addr, void *stream_ptr);

  void CalConv2dPadGradNHWC(const ConvolutionArgs &conv_args, const void *padded_addr, void *dx_addr, void *stream_ptr);

  bool LaunchForward(const ConvolutionArgs &conv_args, const void *input_addr, const void *filter_addr,
                     void *output_addr, const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;

  bool LaunchInputGrad(const ConvolutionArgs &conv_args, const void *dy_addr, const void *filter_addr, void *dx_addr,
                       const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;

  bool LaunchFilterGrad(const ConvolutionArgs &conv_args, const void *dy_addr, const void *input_addr, void *dw_addr,
                        const std::vector<KernelTensor *> &workspace, void *stream_ptr) override;

  void CalInputGradPadList(ConvolutionArgs *conv_args, const ShapeVector &input_shape, const ShapeVector &filter_shape,
                           int h_index, int w_index) {
    if (conv_args->pad_list[kTop2DPadIndex] == -1 || conv_args->pad_list[kBottom2DPadIndex] == -1) {
      int pad_needed_h =
        (static_cast<int>(std::ceil((input_shape[h_index] * 1.0) / conv_args->stride[kHeight2DStrideIndex])) - 1) *
          conv_args->stride[kHeight2DStrideIndex] +
        conv_args->dilation[h_index] * (filter_shape[h_index] - 1) + 1 - input_shape[h_index];
      auto pad_needed_h_final = std::max(0, pad_needed_h);
      conv_args->pad_list[kTop2DPadIndex] =
        static_cast<int>(std::floor(pad_needed_h_final * 1.0 / kConv2dSymmetricCoef));
      conv_args->pad_list[kBottom2DPadIndex] = pad_needed_h_final - conv_args->pad_list[kTop2DPadIndex];
    }
    if (conv_args->pad_list[kLeft2DPadIndex] == -1 || conv_args->pad_list[kRight2DPadIndex] == -1) {
      int pad_needed_w =
        (static_cast<int>(std::ceil((input_shape[w_index] * 1.0) / conv_args->stride[kWidth2DStrideIndex])) - 1) *
          conv_args->stride[kWidth2DStrideIndex] +
        conv_args->dilation[w_index] * (filter_shape[w_index] - 1) + 1 - input_shape[w_index];
      auto pad_needed_w_final = std::max(0, pad_needed_w);
      conv_args->pad_list[kLeft2DPadIndex] =
        static_cast<int>(std::floor(pad_needed_w_final * 1.0 / kConv2dSymmetricCoef));
      conv_args->pad_list[kRight2DPadIndex] = pad_needed_w_final - conv_args->pad_list[kLeft2DPadIndex];
    }
  }

  void CalFilterGradPadList(ConvolutionArgs *conv_args, const ShapeVector &input_shape, const ShapeVector &filter_shape,
                            int h_index, int w_index) {
    auto pad_list = conv_args->pad_list;
    if (pad_list[kTop2DPadIndex] == -1 || pad_list[kBottom2DPadIndex] == -1) {
      int pad_needed_h = (static_cast<int>(std::ceil((input_shape[h_index] * 1.0) / conv_args->stride[kIndex2])) - 1) *
                           conv_args->stride[kIndex2] +
                         conv_args->dilation[kIndex2] * (filter_shape[h_index] - 1) + 1 - input_shape[h_index];
      conv_args->pad_height = std::max(0, pad_needed_h);
      conv_args->pad_top = static_cast<int>(std::floor(conv_args->pad_height * 1.0 / kConv2dSymmetricCoef));
    } else {
      conv_args->pad_height = pad_list[kTop2DPadIndex] + pad_list[kBottom2DPadIndex];
      conv_args->pad_top = pad_list[kTop2DPadIndex];
    }
    if (pad_list[kLeft2DPadIndex] == -1 || pad_list[kRight2DPadIndex] == -1) {
      int pad_needed_w = (static_cast<int>(std::ceil((input_shape[w_index] * 1.0) / conv_args->stride[kIndex3])) - 1) *
                           conv_args->stride[kIndex3] +
                         conv_args->dilation[kIndex3] * (filter_shape[w_index] - 1) + 1 - input_shape[w_index];
      conv_args->pad_width = std::max(0, pad_needed_w);
      conv_args->pad_left = static_cast<int>(std::floor(conv_args->pad_width * 1.0 / kConv2dSymmetricCoef));
    } else {
      conv_args->pad_width = pad_list[kLeft2DPadIndex] + pad_list[kRight2DPadIndex];
      conv_args->pad_left = pad_list[kLeft2DPadIndex];
    }
  }

  int InitialForward(ConvolutionArgs *conv_args, const std::vector<int64_t> &input_shape,
                     const std::vector<int64_t> &filter_shape, const std::vector<int64_t> &output_shape,
                     std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                     std::vector<size_t> *workspace_size_list) override {
    if (conv_args->stride[0] != 1 || conv_args->stride[kIndex1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                        << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                        << "stride[0]: " << conv_args->stride[0] << ", stride[kIndex1]: " << conv_args->stride[kIndex1];
    }
    if (conv_args->dilation[0] != 1 || conv_args->dilation[kIndex1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                        << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << conv_args->dilation[0]
                        << ", dilation[kIndex1]: " << conv_args->dilation[kIndex1];
    }
    SetCudnnComputeInfo(conv_args);

    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    Set4DDesc(conv_args, input_shape, output_shape, filter_shape);

    conv_args->pad_height = conv_args->pad_list[kTop2DPadIndex];
    conv_args->pad_width = conv_args->pad_list[kLeft2DPadIndex];
    conv_args->use_pad = !((conv_args->pad_height == conv_args->pad_list[kBottom2DPadIndex]) &&
                           (conv_args->pad_width == conv_args->pad_list[kRight2DPadIndex]));
    cudnnTensorDescriptor_t real_desc = nullptr;
    int padA[kConv2dDimSize];
    int strideA[kConv2dDimSize] = {conv_args->stride[kHeight2DStrideIndex], conv_args->stride[kWidth2DStrideIndex]};
    int dilaA[kConv2dDimSize] = {conv_args->dilation[kHeight2DDilationIndex],
                                 conv_args->dilation[kWidth2DDilationIndex]};
    if (conv_args->use_pad) {
      conv_args->pad_height = conv_args->pad_list[kTop2DPadIndex] + conv_args->pad_list[kBottom2DPadIndex];
      conv_args->pad_width = conv_args->pad_list[kLeft2DPadIndex] + conv_args->pad_list[kRight2DPadIndex];
      conv_args->pad_top = conv_args->pad_list[kTop2DPadIndex];
      conv_args->pad_left = conv_args->pad_list[kLeft2DPadIndex];

      SetCudnnPaddingInfo(conv_args, padA, strideA, dilaA);
      real_desc = padded_desc_;
    } else {
      if (conv_args->pad_mode == kValidPadModeUpperCase || conv_args->pad_mode == kValidPadModeLowerCase) {
        conv_args->pad_height = 0;
        conv_args->pad_width = 0;
      }
      padA[0] = conv_args->pad_height;
      padA[kIndex1] = conv_args->pad_width;
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      real_desc = tensor0_desc_;
    }

    SetConvolutionMathType(conv_desc_, cudnn_data_type_);
    forward_algo_ = SelectForwardAlgorithm(cudnn_handle_, cudnn_data_type_, real_desc, filter_desc_, conv_desc_,
                                           tensor1_desc_, conv_args->group);
    InitSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
    return KRET_OK;
  }

  int InitialInputGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &dy_shape,
                       const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
                       std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                       std::vector<size_t> *workspace_size_list) override {
    if (conv_args->data_format == kOpFormat_NCHW) {
      if (conv_args->dilation[kIndex0] != 1 || conv_args->dilation[kIndex1] != 1) {
        MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                          << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                          << "dilation[0]: " << conv_args->dilation[kIndex0]
                          << ", dilation[kIndex1]: " << conv_args->dilation[kIndex1];
      }
    } else if (conv_args->data_format == kOpFormat_NHWC) {
      if (conv_args->dilation[kIndex0] != 1 || conv_args->dilation[kIndex3] != 1) {
        MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                          << "', the value of 'dilation' at 0 and 3 axis must be 1, but got "
                          << "dilation[0]: " << conv_args->dilation[kIndex0]
                          << ", dilation[kIndex3]: " << conv_args->dilation[kIndex3];
      }
    }
    auto iter = kFormatIndexMap.find(conv_args->data_format_attr);
    if (iter == kFormatIndexMap.end()) {
      MS_LOG(EXCEPTION) << "OriFormat is " << conv_args->data_format << ", Please confirm that in {NCHW, HWCN, NHWC}.";
    }

    auto stride_ori = conv_args->stride;
    size_t h_ori_index = iter->second;
    const size_t offset = 2;
    int index = 0;
    for (size_t i = h_ori_index; i < h_ori_index + offset; i++) {
      conv_args->stride[index] = static_cast<int>(stride_ori[i]);
      index++;
    }

    SetCudnnComputeInfo(conv_args);
    int h_index = k2DHeightIndexNCHW;
    int w_index = k2DHeightIndexNCHW + 1;
    if (conv_args->data_format == kOpFormat_NHWC) {
      cudnn_compute_format_ = CUDNN_TENSOR_NHWC;
      h_index = k2DHeightIndexNHWC;
      w_index = k2DHeightIndexNHWC + 1;
    }
    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    Set4DDesc(conv_args, dy_shape, input_shape, filter_shape);
    CalInputGradPadList(conv_args, input_shape, filter_shape, h_index, w_index);

    conv_args->pad_height = conv_args->pad_list[kTop2DPadIndex];
    conv_args->pad_width = conv_args->pad_list[kLeft2DPadIndex];
    conv_args->use_pad = !((conv_args->pad_height == conv_args->pad_list[kBottom2DPadIndex]) &&
                           (conv_args->pad_width == conv_args->pad_list[kRight2DPadIndex]));

    cudnnTensorDescriptor_t real_desc = nullptr;
    int padA[kConv2dDimSize];
    int strideA[kConv2dDimSize] = {conv_args->stride[0], conv_args->stride[kIndex1]};
    int dilaA[kConv2dDimSize];
    if (conv_args->data_format == kOpFormat_NHWC) {
      dilaA[kIndex0] = conv_args->dilation[kHeight2DDilationIndex - 1];
      dilaA[kIndex1] = conv_args->dilation[kWidth2DDilationIndex - 1];
    } else {
      dilaA[kIndex0] = conv_args->dilation[kHeight2DDilationIndex];
      dilaA[kIndex1] = conv_args->dilation[kWidth2DDilationIndex];
    }
    if (conv_args->use_pad) {
      conv_args->pad_height = conv_args->pad_list[kTop2DPadIndex] + conv_args->pad_list[kBottom2DPadIndex];
      conv_args->pad_width = conv_args->pad_list[kLeft2DPadIndex] + conv_args->pad_list[kRight2DPadIndex];
      conv_args->pad_top = conv_args->pad_list[kTop2DPadIndex];
      conv_args->pad_left = conv_args->pad_list[kLeft2DPadIndex];
      if (conv_args->pad_height % kConv2dSymmetricCoef == 0 && conv_args->pad_width % kConv2dSymmetricCoef == 0) {
        conv_args->use_pad = false;
      }

      SetCudnnPaddingInfo(conv_args, padA, strideA, dilaA);
      real_desc = padded_desc_;
    } else {
      if (conv_args->pad_mode == kValidPadModeUpperCase || conv_args->pad_mode == kValidPadModeLowerCase) {
        conv_args->pad_height = 0;
        conv_args->pad_width = 0;
      }
      padA[0] = conv_args->pad_height;
      padA[kIndex1] = conv_args->pad_width;
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      real_desc = tensor1_desc_;
    }

    SetConvolutionMathType(conv_desc_, cudnn_data_type_);
    input_grad_algo_ = SelectBackwardDataAlgorithm(cudnn_handle_, cudnn_data_type_, filter_desc_, tensor0_desc_,
                                                   conv_desc_, real_desc, conv_args->group);
    InitSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
    return KRET_OK;
  }

  int InitialFilterGrad(ConvolutionArgs *conv_args, const std::vector<int64_t> &dy_shape,
                        const std::vector<int64_t> &input_shape, const std::vector<int64_t> &filter_shape,
                        std::vector<size_t> *input_size_list, std::vector<size_t> *output_size_list,
                        std::vector<size_t> *workspace_size_list) override {
    if (conv_args->stride[0] != 1 || conv_args->stride[kIndex1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                        << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                        << "stride[0]: " << conv_args->stride[0] << ", stride[kIndex1]: " << conv_args->stride[kIndex1];
    }
    if (conv_args->dilation[0] != 1 || conv_args->dilation[kIndex1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << conv_args->kernel_name
                        << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << conv_args->dilation[0]
                        << ", dilation[kIndex1]: " << conv_args->dilation[kIndex1];
    }
    SetCudnnComputeInfo(conv_args);
    int h_index = k2DHeightIndexNCHW;
    int w_index = k2DHeightIndexNCHW + 1;
    if (conv_args->data_format == kOpFormat_NHWC) {
      cudnn_compute_format_ = CUDNN_TENSOR_NHWC;
      h_index = k2DHeightIndexNHWC;
      w_index = k2DHeightIndexNHWC + 1;
    }
    SetNCHW(input_shape, &conv_args->batch_size, &conv_args->in_channel, &conv_args->in_height, &conv_args->in_width,
            conv_args->data_format);
    Set4DDesc(conv_args, dy_shape, input_shape, filter_shape);
    CalFilterGradPadList(conv_args, input_shape, filter_shape, h_index, w_index);

    conv_args->use_pad =
      !(conv_args->pad_height % kConv2dSymmetricCoef == 0 && conv_args->pad_width % kConv2dSymmetricCoef == 0);

    cudnnTensorDescriptor_t real_desc = nullptr;
    int padA[kConv2dDimSize];
    int strideA[kConv2dDimSize] = {conv_args->stride[kHeight2DStrideIndex], conv_args->stride[kWidth2DStrideIndex]};
    int dilaA[kConv2dDimSize] = {conv_args->dilation[kHeight2DDilationIndex],
                                 conv_args->dilation[kWidth2DDilationIndex]};
    if (conv_args->use_pad) {
      SetCudnnPaddingInfo(conv_args, padA, strideA, dilaA);

      real_desc = padded_desc_;
    } else {
      if (conv_args->pad_mode == kValidPadModeUpperCase || conv_args->pad_mode == kValidPadModeLowerCase) {
        conv_args->pad_top = 0;
        conv_args->pad_left = 0;
      }
      padA[0] = conv_args->pad_top;
      padA[kIndex1] = conv_args->pad_left;
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                        CUDNN_DATA_FLOAT),
        "cudnnSetConvolution2dDescriptor failed");
      real_desc = tensor1_desc_;
    }

    SetConvolutionMathType(conv_desc_, cudnn_data_type_);
    filter_grad_algo_ = SelectBackwardFilterAlgorithm(cudnn_handle_, cudnn_data_type_, real_desc, tensor0_desc_,
                                                      conv_desc_, filter_desc_, conv_args->group);

    InitSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
    return KRET_OK;
  }

  void SetCudnnComputeInfo(ConvolutionArgs *conv_args) {
    cudnn_data_type_ = GetCudnnDataType(conv_args->data_type);
    if (conv_args->data_format == kOpFormat_NHWC) {
      cudnn_compute_format_ = CUDNN_TENSOR_NHWC;
    } else if (conv_args->data_format == kOpFormat_NCHW) {
      cudnn_compute_format_ = CUDNN_TENSOR_NCHW;
    } else {
      MS_LOG(EXCEPTION) << "Data format:" << conv_args->data_format << " is invalid.";
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionGroupCount(conv_desc_, conv_args->group),
                                        "cudnnSetConvGroupCount failed");
  }

  void SetCudnnPaddingInfo(ConvolutionArgs *conv_args, int *padA, int *strideA, int *dilaA) {
    int dimA[kConv2dInputDimSize];
    int strideAPadded[kConv2dInputDimSize];
    ShapeVector padded_shape;
    if (conv_args->data_format == kOpFormat_NCHW || conv_args->data_format == kOpFormat_DEFAULT) {
      padded_shape = {conv_args->batch_size, conv_args->in_channel, conv_args->in_height + conv_args->pad_height,
                      conv_args->in_width + conv_args->pad_width};
      SetDimA(padded_shape, dimA, kConv2dInputDimSize, conv_args->data_format);
      SetStrideA(padded_shape, strideAPadded, kConv2dInputDimSize, conv_args->data_format);
    } else if (conv_args->data_format == kOpFormat_NHWC) {
      padded_shape = {conv_args->batch_size, conv_args->in_height + conv_args->pad_height,
                      conv_args->in_width + conv_args->pad_width, conv_args->in_channel};
      SetDimA(padded_shape, dimA, kConv2dInputDimSize, conv_args->data_format);
      SetStrideA(padded_shape, strideAPadded, kConv2dInputDimSize, conv_args->data_format);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, kConv2dInputDimSize, dimA, strideAPadded),
      "cudnnSetTensor4dDescriptor failed");
    padA[0] = 0;
    padA[kIndex1] = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolutionNdDescriptor failed");
  }

  void InitForwardSizeLists(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                            std::vector<size_t> *output_size_list, std::vector<size_t> *workspace_size_list) {
    input_size_list->push_back(tensor0_size_);
    input_size_list->push_back(filter_size_);
    output_size_list->push_back(tensor1_size_);
    if (conv_args->use_pad) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, tensor1_desc_,
                                                forward_algo_, &workspace_size_),
        GetConvForwardInfo("cudnnGetConvolutionForwardWorkspaceSize failed", padded_desc_, filter_desc_, conv_desc_,
                           tensor1_desc_));
      workspace_size_list->push_back(padded_size_);
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, tensor0_desc_, filter_desc_, conv_desc_, tensor1_desc_,
                                                forward_algo_, &workspace_size_),
        GetConvForwardInfo("cudnnGetConvolutionForwardWorkspaceSize failed", tensor0_desc_, filter_desc_, conv_desc_,
                           tensor1_desc_));
    }
    workspace_size_list->insert(workspace_size_list->begin(), workspace_size_);
  }

  void InitInputGradSizeLists(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                              std::vector<size_t> *output_size_list, std::vector<size_t> *workspace_size_list) {
    input_size_list->push_back(tensor0_size_);
    input_size_list->push_back(filter_size_);
    output_size_list->push_back(tensor1_size_);

    if (conv_args->use_pad) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, tensor0_desc_, conv_desc_,
                                                     padded_desc_, input_grad_algo_, &workspace_size_),
        GetConvBwdDataInfo("cudnnGetConvolutionBackwardDataWorkspaceSize failed", filter_desc_, tensor0_desc_,
                           conv_desc_, padded_desc_));
      workspace_size_list->push_back(padded_size_);
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, filter_desc_, tensor0_desc_, conv_desc_,
                                                     tensor1_desc_, input_grad_algo_, &workspace_size_),
        GetConvBwdDataInfo("cudnnGetConvolutionBackwardDataWorkspaceSize failed", filter_desc_, tensor0_desc_,
                           conv_desc_, tensor1_desc_));
    }
    workspace_size_list->insert(workspace_size_list->begin(), workspace_size_);
  }

  void InitFilterGradSizeLists(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                               std::vector<size_t> *output_size_list, std::vector<size_t> *workspace_size_list) {
    input_size_list->push_back(tensor0_size_);
    input_size_list->push_back(tensor1_size_);
    output_size_list->push_back(filter_size_);

    if (conv_args->use_pad) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, padded_desc_, tensor0_desc_, conv_desc_,
                                                       filter_desc_, filter_grad_algo_, &workspace_size_),
        GetConvBwdFilterInfo("cudnnGetConvolutionBackwardFilterWorkspaceSize failed", padded_desc_, tensor0_desc_,
                             conv_desc_, filter_desc_));
      workspace_size_list->push_back(padded_size_);
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, tensor1_desc_, tensor0_desc_, conv_desc_,
                                                       filter_desc_, filter_grad_algo_, &workspace_size_),
        GetConvBwdFilterInfo("cudnnGetConvolutionBackwardFilterWorkspaceSize failed", tensor1_desc_, tensor0_desc_,
                             conv_desc_, filter_desc_));
    }
    workspace_size_list->insert(workspace_size_list->begin(), workspace_size_);
  }

  void InitSizeLists(ConvolutionArgs *conv_args, std::vector<size_t> *input_size_list,
                     std::vector<size_t> *output_size_list, std::vector<size_t> *workspace_size_list) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(tensor0_desc_, reinterpret_cast<size_t *>(&tensor0_size_)),
      "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(tensor1_desc_, reinterpret_cast<size_t *>(&tensor1_size_)),
      "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)),
      "cudnnGetFilterSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_ERROR_NOTRACE(
      cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
      "cudnnGetTensorSizeInBytes failed");
    auto conv_type = get_conv_type();
    switch (conv_type) {
      case ConvType::kForward:
        InitForwardSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
        break;
      case ConvType::kInputGrad:
        InitInputGradSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
        break;
      case ConvType::kFilterGrad:
        InitFilterGradSizeLists(conv_args, input_size_list, output_size_list, workspace_size_list);
        break;
      default:
        MS_LOG(EXCEPTION) << "Convolution type: " << conv_type << " is invalid.";
    }
  }

  void Set4DDesc(ConvolutionArgs *conv_args, const ShapeVector &tensor0_shape, const ShapeVector &tensor1_shape,
                 const ShapeVector &filter_shape) {
    auto conv_type = get_conv_type();
    int dimA0[kConv2dInputDimSize];
    int strideA0[kConv2dInputDimSize];
    int dimA1[kConv2dInputDimSize];
    int strideA1[kConv2dInputDimSize];
    int filterDimA[kConv2dInputDimSize];

    SetDimA(tensor0_shape, dimA0, kConv2dInputDimSize, conv_args->data_format);
    SetStrideA(tensor0_shape, strideA0, kConv2dInputDimSize, conv_args->data_format);

    SetDimA(tensor1_shape, dimA1, kConv2dInputDimSize, conv_args->data_format);
    SetStrideA(tensor1_shape, strideA1, kConv2dInputDimSize, conv_args->data_format);

    if (conv_type == ConvType::kFilterGrad) {
      // filter shape relued by format_attr_. In native mode it's OHWI. In transpose mode it's OIHW.
      SetDimA(filter_shape, filterDimA, kConv2dInputDimSize, conv_args->data_format_attr);
    } else {
      SetDimA(filter_shape, filterDimA, kConv2dInputDimSize, conv_args->data_format);
    }

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(tensor0_desc_, cudnn_data_type_, kConv2dInputDimSize, dimA0, strideA0),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(tensor1_desc_, cudnn_data_type_, kConv2dInputDimSize, dimA1, strideA1),
      "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, cudnn_compute_format_, kConv2dInputDimSize,
                                 filterDimA),
      "cudnnSetFilterNdDescriptor failed");
  }

  cudnnHandle_t cudnn_handle_{nullptr};
  cudnnConvolutionDescriptor_t conv_desc_{nullptr};
  cudnnTensorDescriptor_t tensor0_desc_{nullptr};
  cudnnTensorDescriptor_t tensor1_desc_{nullptr};
  cudnnFilterDescriptor_t filter_desc_{nullptr};
  cudnnTensorDescriptor_t padded_desc_{nullptr};

  cudnnConvolutionFwdAlgo_t forward_algo_{CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM};
  cudnnConvolutionBwdDataAlgo_t input_grad_algo_{CUDNN_CONVOLUTION_BWD_DATA_ALGO_1};
  cudnnConvolutionBwdFilterAlgo_t filter_grad_algo_{CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1};

  cudnnDataType_t cudnn_data_type_{CUDNN_DATA_FLOAT};
  cudnnTensorFormat_t cudnn_compute_format_{CUDNN_TENSOR_NCHW};

  size_t tensor0_size_{0};
  size_t tensor1_size_{0};
  size_t filter_size_{0};
  size_t padded_size_{0};
  size_t workspace_size_{0};
  std::string kernel_name_;
};

template <typename T>
void ConvolutionCudnnGpuKernel<T>::CalConv2dPadNCHW(const ConvolutionArgs &conv_args, const void *input_addr,
                                                    void *padded_addr, void *stream_ptr) {
  const float pad_value{0.0};
  CalPad(padded_size_ / sizeof(T), static_cast<T *>(const_cast<void *>(input_addr)), conv_args.batch_size,
         conv_args.in_channel, conv_args.in_height, conv_args.in_width, conv_args.in_height + conv_args.pad_height,
         conv_args.in_width + conv_args.pad_width, conv_args.pad_top, conv_args.pad_left, pad_value,
         static_cast<T *>(padded_addr), reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename T>
void ConvolutionCudnnGpuKernel<T>::CalConv2dPadNHWC(const ConvolutionArgs &conv_args, const void *input_addr,
                                                    void *padded_addr, void *stream_ptr) {
  const float pad_value{0.0};
  CalPadNHWC(padded_size_ / sizeof(T), static_cast<T *>(const_cast<void *>(input_addr)), conv_args.batch_size,
             conv_args.in_height, conv_args.in_width, conv_args.in_channel, conv_args.in_height + conv_args.pad_height,
             conv_args.in_width + conv_args.pad_width, conv_args.pad_top, conv_args.pad_left, pad_value,
             static_cast<T *>(padded_addr), reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename T>
void ConvolutionCudnnGpuKernel<T>::CalConv2dPadGradNCHW(const ConvolutionArgs &conv_args, const void *padded_addr,
                                                        void *dx_addr, void *stream_ptr) {
  CalPadGrad(tensor1_size_ / sizeof(T), static_cast<T *>(const_cast<void *>(padded_addr)), conv_args.batch_size,
             conv_args.in_channel, conv_args.in_height, conv_args.in_width, conv_args.in_height + conv_args.pad_height,
             conv_args.in_width + conv_args.pad_width, conv_args.pad_top, conv_args.pad_left, static_cast<T *>(dx_addr),
             reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename T>
void ConvolutionCudnnGpuKernel<T>::CalConv2dPadGradNHWC(const ConvolutionArgs &conv_args, const void *padded_addr,
                                                        void *dx_addr, void *stream_ptr) {
  CalPadGradNHWC(tensor1_size_ / sizeof(T), static_cast<T *>(const_cast<void *>(padded_addr)), conv_args.batch_size,
                 conv_args.in_height, conv_args.in_width, conv_args.in_channel,
                 conv_args.in_height + conv_args.pad_height, conv_args.in_width + conv_args.pad_width,
                 conv_args.pad_top, conv_args.pad_left, static_cast<T *>(dx_addr),
                 reinterpret_cast<cudaStream_t>(stream_ptr));
}

template <typename T>
bool ConvolutionCudnnGpuKernel<T>::LaunchForward(const ConvolutionArgs &conv_args, const void *input_addr,
                                                 const void *filter_addr, void *output_addr,
                                                 const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  void *workspace_addr = nullptr;
  void *padded_addr = nullptr;
  if (workspace.size() >= 1) {
    workspace_addr = GetPossiblyNullDeviceAddress<void>(workspace, 0);
  }
  if (conv_args.use_pad && workspace.size() >= kIndex2) {
    padded_addr = GetDeviceAddress<void>(workspace, kIndex1);
  }
  if (conv_args.use_pad) {
    if (conv_args.data_format == kOpFormat_NHWC) {
      CalConv2dPadNHWC(conv_args, input_addr, padded_addr, stream_ptr);
    } else {
      CalConv2dPadNCHW(conv_args, input_addr, padded_addr, stream_ptr);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionForward(cudnn_handle_, &conv_args.alpha, padded_desc_, padded_addr, filter_desc_, filter_addr,
                              conv_desc_, forward_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                              tensor1_desc_, output_addr),
      "cudnnConvolutionForward failed");
  } else {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionForward(cudnn_handle_, &conv_args.alpha, tensor0_desc_, input_addr, filter_desc_, filter_addr,
                              conv_desc_, forward_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                              tensor1_desc_, output_addr),
      "cudnnConvolutionForward failed");
  }
  return true;
}

template <typename T>
bool ConvolutionCudnnGpuKernel<T>::LaunchInputGrad(const ConvolutionArgs &conv_args, const void *dy_addr,
                                                   const void *filter_addr, void *dx_addr,
                                                   const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  void *workspace_addr = nullptr;
  void *padded_addr = nullptr;
  if (workspace.size() >= 1) {
    workspace_addr = GetPossiblyNullDeviceAddress<void>(workspace, 0);
  }
  if (conv_args.use_pad && workspace.size() >= kIndex2) {
    padded_addr = GetDeviceAddress<void>(workspace, kIndex1);
  }
  if (conv_args.use_pad) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardData(cudnn_handle_, &conv_args.alpha, filter_desc_, filter_addr, tensor0_desc_, dy_addr,
                                   conv_desc_, input_grad_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                                   padded_desc_, padded_addr),
      "ConvolutionBackwardData failed");
    if (conv_args.data_format == kOpFormat_NHWC) {
      CalConv2dPadGradNHWC(conv_args, padded_addr, dx_addr, stream_ptr);
    } else {
      CalConv2dPadGradNCHW(conv_args, padded_addr, dx_addr, stream_ptr);
    }
  } else {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardData(cudnn_handle_, &conv_args.alpha, filter_desc_, filter_addr, tensor0_desc_, dy_addr,
                                   conv_desc_, input_grad_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                                   tensor1_desc_, dx_addr),
      "ConvolutionBackwardData failed");
  }
  return true;
}

template <typename T>
bool ConvolutionCudnnGpuKernel<T>::LaunchFilterGrad(const ConvolutionArgs &conv_args, const void *dy_addr,
                                                    const void *input_addr, void *dw_addr,
                                                    const std::vector<KernelTensor *> &workspace, void *stream_ptr) {
  void *workspace_addr = nullptr;
  void *padded_addr = nullptr;
  if (workspace.size() >= 1) {
    workspace_addr = GetPossiblyNullDeviceAddress<void>(workspace, 0);
  }
  if (conv_args.use_pad && workspace.size() >= kIndex2) {
    padded_addr = GetDeviceAddress<void>(workspace, kIndex1);
  }
  if (conv_args.use_pad) {
    if (conv_args.data_format == kOpFormat_NHWC) {
      CalConv2dPadNHWC(conv_args, input_addr, padded_addr, stream_ptr);
    } else {
      CalConv2dPadNCHW(conv_args, input_addr, padded_addr, stream_ptr);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardFilter(cudnn_handle_, &conv_args.alpha, padded_desc_, padded_addr, tensor0_desc_, dy_addr,
                                     conv_desc_, filter_grad_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                                     filter_desc_, dw_addr),
      "ConvolutionBackwardFilter failed");
  } else {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardFilter(cudnn_handle_, &conv_args.alpha, tensor1_desc_, input_addr, tensor0_desc_, dy_addr,
                                     conv_desc_, filter_grad_algo_, workspace_addr, workspace_size_, &conv_args.beta,
                                     filter_desc_, dw_addr),
      "ConvolutionBackwardFilter failed");
  }
  return true;
}
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_CONV2D_CUDNN_GPU_KERNEL_H_
