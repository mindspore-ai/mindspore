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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_INPUT_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_INPUT_GPU_KERNEL_H_

#include <algorithm>
#include <string>
#include <vector>

#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/kernel_constants.h"

namespace mindspore {
namespace kernel {
constexpr int kNumDims = 5;
constexpr int kConvDims = 3;

constexpr size_t k3DPadSize = 6;
constexpr size_t kHead3DPadIndex = 0;
constexpr size_t kTail3DPadIndex = 1;
constexpr size_t kTop3DPadIndex = 2;
constexpr size_t kBottom3DPadIndex = 3;
constexpr size_t kLeft3DPadIndex = 4;
constexpr size_t kRight3DPadIndex = 5;

constexpr size_t kPadIdxDepth = 0;
constexpr size_t kPadIdxHeight = 1;
constexpr size_t kPadIdxWidth = 2;

constexpr size_t k3DStrideSize = 5;
constexpr size_t kDepth3DStrideIndex = 2;
constexpr size_t kHeight3DStrideIndex = 3;
constexpr size_t kWidth3DStrideIndex = 4;

constexpr size_t k3DDilationSize = 5;
constexpr size_t kDepth3DDilationIndex = 2;
constexpr size_t kHeight3DDilationIndex = 3;
constexpr size_t kWidth3DDilationIndex = 4;

template <typename T>
class Conv3dGradInputGpuKernelMod : public DeprecatedNativeGpuKernelMod {
 public:
  Conv3dGradInputGpuKernelMod() { ResetResource(); }
  ~Conv3dGradInputGpuKernelMod() override { DestroyResource(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    T *w = GetDeviceAddress<T>(inputs, 0);
    T *dy = GetDeviceAddress<T>(inputs, 1);
    T *dx = GetDeviceAddress<T>(outputs, 0);
    T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

    const float alpha = 1;
    if (use_pad_) {
      T *padded = GetDeviceAddress<T>(workspace, 1);

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta_, padded_descriptor_, padded),
        "ConvolutionBackwardData failed");
      CalPadGrad3d(output_size_ / sizeof(T), padded, n_, c_, old_depth_, old_height_, old_width_,
                   old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_, pad_head_, pad_top_,
                   pad_left_, dx, reinterpret_cast<cudaStream_t>(stream_ptr));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, w_desc_, w, dy_desc_, dy, conv_desc_, algo_, work_space,
                                     workspace_size_, &beta_, dx_desc_, dx),
        "ConvolutionBackwardData failed");
    }
    return true;
  }

  void CheckSize(const size_t value, const size_t expect_value, const string arg_name) {
    if (value != expect_value) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of " << arg_name << " must be " << expect_value
                        << ", but got " << value;
    }
  }

  bool Init(const CNodePtr &kernel_node) override {
    kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
    kernel_node_ = kernel_node;
    InitResource();
    (void)CheckParam(kernel_node);
    cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(AnfAlgo::GetInputDeviceDataType(kernel_node, 0)));
    data_format_ = kOpFormat_NCDHW;
    auto filter_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
    auto dy_shape = AnfAlgo::GetInputDeviceShape(kernel_node, 1);
    is_null_input_ =
      CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") || CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy");
    if (is_null_input_ || AnfAlgo::IsShapesDynamic({filter_shape, dy_shape})) {
      InitSizeLists();
      return true;
    }
    ShapeVector input_shape;
    GetInputShape(kernel_node, &input_shape);
    compute_format_ = CUDNN_TENSOR_NCHW;
    CheckTensorSize({input_shape});
    (void)CheckSize(input_shape.size(), 5, "input");
    n_ = LongToInt(input_shape[0]);
    c_ = LongToInt(input_shape[1]);
    old_depth_ = LongToInt(input_shape[2]);
    old_height_ = LongToInt(input_shape[3]);
    old_width_ = LongToInt(input_shape[4]);
    SetNDDesc(dy_shape, input_shape, filter_shape);
    group_ = static_cast<int>(GetAttr<int64_t>(kernel_node, "group"));
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                "cudnnSetConvGroupCount failed");
    std::vector<int> pad_list;
    std::vector<int64_t> pad_list_me = GetAttr<std::vector<int64_t>>(kernel_node, "pad_list");
    (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                         [](const int64_t &value) { return static_cast<int>(value); });
    SetPad(kernel_node, pad_list);
    SetStrideAndDilation(kernel_node);
    auto dx_desc_real = GetDxDescReal(pad_list);
    if (cudnn_data_type_ == CUDNN_DATA_HALF) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                  "cudnnSetConvolutionMathType failed.")
    }
    SelectAlgorithm(dx_desc_real);
    beta_ = GetAttrWithDefault(kernel_node, "inplace_algo", std::string("cover")) == "cover" ? 0 : 1;
    InitSizeLists();
    return true;
  }

  void ResetResource() noexcept override {
    cudnn_handle_ = nullptr;
    w_desc_ = nullptr;
    conv_desc_ = nullptr;
    dy_desc_ = nullptr;
    dx_desc_ = nullptr;
    padded_descriptor_ = nullptr;
    algo_ = CUDNN_CONVOLUTION_BWD_DATA_ALGO_0;
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
    group_ = 1;
    is_null_input_ = false;
    kernel_name_ = "Conv3dGradInput";
    dy_size_ = 0;
    w_size_ = 0;
    output_size_ = 0;
    padded_size_ = 0;
    workspace_size_ = 0;
    use_pad_ = true;
    beta_ = 0;
  }

  void DestroyResource() noexcept override {
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyConvolutionDescriptor(conv_desc_),
                               "cudnnDestroyConvolutionDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyFilterDescriptor(w_desc_),
                               "cudnnDestroyFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(padded_descriptor_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dy_desc_),
                               "cudnnDestroyTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_ERROR(kernel_node_, cudnnDestroyTensorDescriptor(dx_desc_),
                               "cudnnDestroyTensorDescriptor failed");
  }

 protected:
  void InitResource() override {
    cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dx_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&dy_desc_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateTensorDescriptor(&padded_descriptor_),
                                "cudnnCreateTensorDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateFilterDescriptor(&w_desc_),
                                "cudnnCreateFilterDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnCreateConvolutionDescriptor(&conv_desc_),
                                "cudnnCreateConvolutionDescriptor failed");
  }

  void InitSizeLists() override {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dy_desc_, &dy_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetFilterSizeInBytes(w_desc_, &w_size_),
                                  "cudnnGetTensorSizeInBytes failed");
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(dx_desc_, &output_size_),
                                  "cudnnGetTensorSizeInBytes failed");
    }

    input_size_list_.push_back(dy_size_);
    input_size_list_.push_back(w_size_);
    output_size_list_.push_back(output_size_);

    if (use_pad_ && !is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_, cudnnGetTensorSizeInBytes(padded_descriptor_, &padded_size_),
                                  "cudnnGetTensorSizeInBytes failed");

      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_,
        cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, padded_descriptor_,
                                                     algo_, &workspace_size_),
        "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      workspace_size_list_.push_back(padded_size_);
    } else {
      if (!is_null_input_) {
        CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                    cudnnGetConvolutionBackwardDataWorkspaceSize(
                                      cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_, algo_, &workspace_size_),
                                    "cudnnGetConvolutionBackwardDataWorkspaceSize failed");
      }
    }
    (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
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

  void SelectAlgorithm(cudnnTensorDescriptor_t dx_desc_real) {
    const int requested_algo_count = 1;
    int returned_algo_count = 0;
    cudnnConvolutionBwdDataAlgoPerf_t perf_results;
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_,
      cudnnGetConvolutionBackwardDataAlgorithm_v7(cudnn_handle_, w_desc_, dy_desc_, conv_desc_, dx_desc_real,
                                                  requested_algo_count, &returned_algo_count, &perf_results),
      "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
    algo_ = perf_results.algo;
  }

  void GetInputShape(const CNodePtr &kernel_node, ShapeVector *input_shape) {
    auto shp_tuple_x = GetAttrAndConvertValueTuple(kernel_node, "input_size");
    (void)std::transform(std::begin(shp_tuple_x), std::end(shp_tuple_x), std::back_inserter(*input_shape),
                         [](const ValuePtr &e) -> int64_t {
                           auto cast_value = e->cast<Int64ImmPtr>();
                           MS_EXCEPTION_IF_NULL(cast_value);
                           return static_cast<int64_t>(cast_value->value());
                         });
  }

  void SetNDDesc(const ShapeVector &dy_shape, const ShapeVector &input_shape, const ShapeVector &filter_shape) {
    const int kDims = 5;
    int dimA[kDims];
    int strideAin[kDims];
    int dimAdy[kDims];
    int strideAdy[kDims];
    int filterDimA[kDims];
    SetDimA(input_shape, dimA, kDims, data_format_);
    SetStrideA(input_shape, strideAin, kDims, data_format_);
    SetDimA(dy_shape, dimAdy, kDims, data_format_);
    SetStrideA(dy_shape, strideAdy, kDims, data_format_);
    SetDimA(filter_shape, filterDimA, kDims, data_format_);

    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, kDims, dimAdy, strideAdy),
                                "cudnnSetTensorNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(
      kernel_node_, cudnnSetFilterNdDescriptor(w_desc_, cudnn_data_type_, compute_format_, kDims, filterDimA),
      "cudnnSetFilterNdDescriptor failed");
    CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                cudnnSetTensorNdDescriptor(dx_desc_, cudnn_data_type_, kDims, dimA, strideAin),
                                "cudnnSetTensorNdDescriptor failed");
  }

  void SetStrideAndDilation(const CNodePtr &kernel_node) {
    std::vector<int64_t> stride_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "strides");
    std::vector<int64_t> dilation_me = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, "dilations");
    (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                         [](const int64_t &value) { return static_cast<int>(value); });
    if (stride_.size() != k3DStrideSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 5, but got " << stride_.size();
    }
    if (stride_[0] != 1 || stride_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                        << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
    }
    if (dilation_.size() != k3DDilationSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 5, but got "
                        << dilation_.size();
    }
    if (dilation_[0] != 1 || dilation_[1] != 1) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                        << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
    }
  }

  void SetPad(const CNodePtr &kernel_node, const std::vector<int> &pad_list) {
    if (pad_list.size() != k3DPadSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 6, but got " << pad_list.size();
    }
    pad_depth_ = pad_list[kHead3DPadIndex];
    pad_height_ = pad_list[kTop3DPadIndex];
    pad_width_ = pad_list[kLeft3DPadIndex];
    use_pad_ = !((pad_depth_ == pad_list[1]) && (pad_height_ == pad_list[3]) && (pad_width_ == pad_list[5]));
    pad_mode_ = GetAttr<std::string>(kernel_node, "pad_mode");
  }

  cudnnTensorDescriptor_t GetDxDescReal(const std::vector<int> &pad_list) {
    cudnnTensorDescriptor_t dx_desc_real = nullptr;
    int padA[kConvDims];
    int strideA[kConvDims] = {stride_[kDepth3DStrideIndex], stride_[kHeight3DStrideIndex],
                              stride_[kWidth3DStrideIndex]};
    int dilaA[kConvDims] = {dilation_[kDepth3DDilationIndex], dilation_[kHeight3DDilationIndex],
                            dilation_[kWidth3DDilationIndex]};
    if (use_pad_) {
      pad_depth_ = pad_list[kHead3DPadIndex] + pad_list[kTail3DPadIndex];
      pad_height_ = pad_list[kTop3DPadIndex] + pad_list[kBottom3DPadIndex];
      pad_width_ = pad_list[kLeft3DPadIndex] + pad_list[kRight3DPadIndex];
      pad_head_ = pad_list[kHead3DPadIndex];
      pad_top_ = pad_list[kTop3DPadIndex];
      pad_left_ = pad_list[kLeft3DPadIndex];
      int dimA[kNumDims];
      int strideApadded[kNumDims];
      if (data_format_ != kOpFormat_NCDHW) {
        MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'data_format' only support 'NCDHW' right now "
                          << ", but got " << data_format_;
      }
      ShapeVector padded_shape = {n_, c_, old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_};
      SetDimA(padded_shape, dimA, kNumDims, data_format_);
      SetStrideA(padded_shape, strideApadded, kNumDims, data_format_);
      CHECK_CUDNN_RET_WITH_EXCEPT(
        kernel_node_, cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, kNumDims, dimA, strideApadded),
        "cudnnSetTensorNdDescriptor failed");
      padA[kPadIdxDepth] = 0;
      padA[kPadIdxHeight] = 0;
      padA[kPadIdxWidth] = 0;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolutionNdDescriptor failed");
      dx_desc_real = padded_descriptor_;
    } else {
      if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
        pad_depth_ = 0;
        pad_height_ = 0;
        pad_width_ = 0;
      }
      padA[kPadIdxDepth] = pad_depth_;
      padA[kPadIdxHeight] = pad_height_;
      padA[kPadIdxWidth] = pad_width_;
      CHECK_CUDNN_RET_WITH_EXCEPT(kernel_node_,
                                  cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                  CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                  "cudnnSetConvolution2dDescriptor failed");
      dx_desc_real = dx_desc_;
    }

    return dx_desc_real;
  }

  cudnnHandle_t cudnn_handle_;
  cudnnFilterDescriptor_t w_desc_;
  cudnnConvolutionDescriptor_t conv_desc_;
  cudnnTensorDescriptor_t dy_desc_;
  cudnnTensorDescriptor_t dx_desc_;
  cudnnTensorDescriptor_t padded_descriptor_;
  cudnnConvolutionBwdDataAlgo_t algo_;
  std::string pad_mode_;
  std::string data_format_ = kOpFormat_NCDHW;

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
  size_t dy_size_;
  size_t w_size_;
  size_t output_size_;
  size_t padded_size_;
  size_t workspace_size_;
  bool use_pad_;
  float beta_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_NN_CONV3D_GRAD_INPUT_GPU_KERNEL_H_
