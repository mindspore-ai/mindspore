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

#include "plugin/device/gpu/kernel/nn/conv2d_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kConv2dDimSize = 2;
constexpr size_t kInputDimSize = 4;
constexpr size_t kTop2DPadIndex = 0;
constexpr size_t kBottom2DPadIndex = 1;
constexpr size_t kLeft2DPadIndex = 2;
constexpr size_t kRight2DPadIndex = 3;

using KernelRunFunc = Conv2dFwdGpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &Conv2dFwdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &Conv2dFwdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &Conv2dFwdGpuKernelMod::LaunchKernel<half>}};
  return func_list;
}

bool Conv2dFwdGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                                   const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool Conv2dFwdGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                         const std::vector<AddressPtr> &workspace,
                                         const std::vector<AddressPtr> &outputs) {
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
                 reinterpret_cast<cudaStream_t>(stream_ptr_));
    } else {
      CalPad(padded_size_ / sizeof(T), input_addr, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
             old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded_addr,
             reinterpret_cast<cudaStream_t>(stream_ptr_));
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionForward(cudnn_handle_, &alpha, padded_desc_, padded_addr, filter_desc_, filter_addr, conv_desc_,
                              conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
      "cudnnConvolutionForward failed");
  } else {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionForward(cudnn_handle_, &alpha, input_desc_, input_addr, filter_desc_, filter_addr, conv_desc_,
                              conv_algorithm_, workspace_addr, workspace_size_, &beta, output_desc_, output_addr),
      "cudnnConvolutionForward failed");
  }
  return true;
}

bool Conv2dFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  InitResource();
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->GetDtype()));
  data_format_ = mindspore::FormatEnumToString(inputs[0]->GetFormat());
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  auto format_attr = GetValue<std::string>(prim->GetAttr("format"));
  if (format_attr == kOpFormat_NHWC) {
    data_format_ = kOpFormat_NHWC;
  }
  if (data_format_ == kOpFormat_NHWC) {
    compute_format_ = CUDNN_TENSOR_NHWC;
  }
  group_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("group")));
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                      "cudnnSetConvGroupCount failed");
  pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
  auto stride_ori = GetValue<std::vector<int64_t>>(prim->GetAttr("stride"));
  auto dilation_ori = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  SetStrideAndDilation(stride_ori, dilation_ori);
  return true;
}

int Conv2dFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  auto in_shape = inputs[0]->GetDeviceShapeAdaptively();
  auto filter_shape = inputs[1]->GetDeviceShapeAdaptively();
  auto output_shape = outputs[0]->GetDeviceShapeAdaptively();
  is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "x") ||
                   CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") ||
                   CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    return KRET_OK;
  }
  CheckTensorSize({in_shape, filter_shape, output_shape});
  std::vector<int> pad_list;
  // The pad_list is computed in infer shape
  auto pad_list_me = GetValue<std::vector<int64_t>>(base_operator->GetAttr("pad_list"));
  (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (pad_list.size() != kInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got " << pad_list.size();
  }
  SetNCHW(in_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
  Set4DDesc(in_shape, filter_shape, output_shape);
  cudnnTensorDescriptor_t input_descriptor_real = nullptr;
  int padA[2];
  int strideA[2] = {stride_[2], stride_[3]};
  int dilaA[2] = {dilation_[2], dilation_[3]};
  pad_height_ = pad_list[kTop2DPadIndex];
  pad_width_ = pad_list[kLeft2DPadIndex];
  use_pad_ = !((pad_height_ == pad_list[kBottom2DPadIndex]) && (pad_width_ == pad_list[kRight2DPadIndex]));
  if (use_pad_) {
    pad_height_ = pad_list[kTop2DPadIndex] + pad_list[kBottom2DPadIndex];
    pad_width_ = pad_list[kLeft2DPadIndex] + pad_list[kRight2DPadIndex];
    pad_top_ = pad_list[kTop2DPadIndex];
    pad_left_ = pad_list[kLeft2DPadIndex];
    int dimA[kInputDimSize];
    int strideApadded[kInputDimSize];
    if (data_format_ == kOpFormat_NCHW || data_format_ == kOpFormat_DEFAULT) {
      ShapeVector padded_shape = {n_, c_, old_height_ + pad_height_, old_width_ + pad_width_};
      SetDimA(padded_shape, dimA, kInputDimSize, data_format_);
      SetStrideA(padded_shape, strideApadded, kInputDimSize, data_format_);
    } else if (data_format_ == kOpFormat_NHWC) {
      ShapeVector padded_shape = {n_, old_height_ + pad_height_, old_width_ + pad_width_, c_};
      SetDimA(padded_shape, dimA, kInputDimSize, data_format_);
      SetStrideA(padded_shape, strideApadded, kInputDimSize, data_format_);
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, kInputDimSize, dimA, strideApadded),
      "cudnnSetTensor4dDescriptor failed");
    padA[0] = 0;
    padA[1] = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolutionNdDescriptor failed");
    input_descriptor_real = padded_desc_;
  } else {
    if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
      pad_height_ = 0;
      pad_width_ = 0;
    }
    padA[0] = pad_height_;
    padA[1] = pad_width_;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolution2dDescriptor failed");
    input_descriptor_real = input_desc_;
  }
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                        "cudnnSetConvolutionMathType failed.")
  }
  SelectAlgorithm(input_descriptor_real);
  InitSizeLists();
  return KRET_OK;
}

void Conv2dFwdGpuKernelMod::SetStrideAndDilation(const std::vector<int64_t> &stride_me,
                                                 const std::vector<int64_t> &dilation_me) {
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (stride_.size() != kInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 4, but got " << stride_.size();
  }
  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                      << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
  }
  if (dilation_.size() != kInputDimSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                      << dilation_.size();
  }
  if (dilation_[0] != 1 || dilation_[1] != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                      << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
  }
}

void Conv2dFwdGpuKernelMod::Set4DDesc(const ShapeVector &in_shape, const ShapeVector &filter_shape,
                                      const ShapeVector &output_shape) {
  int dimA[kInputDimSize];
  int strideAin[kInputDimSize];
  int dimAout[kInputDimSize];
  int strideAout[kInputDimSize];
  SetDimA(in_shape, dimA, kInputDimSize, data_format_);
  SetStrideA(in_shape, strideAin, kInputDimSize, data_format_);
  SetDimA(output_shape, dimAout, kInputDimSize, data_format_);
  SetStrideA(output_shape, strideAout, kInputDimSize, data_format_);
  int filterDimA[4];
  // OHWI for NHWC; OIHW for NCHW
  SetDimA(filter_shape, filterDimA, kInputDimSize, data_format_);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, kInputDimSize, dimA, strideAin),
    "cudnnSetTensor4dDescriptor failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, kInputDimSize, filterDimA),
    "cudnnSetFilter4dDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, kInputDimSize, dimAout, strideAout),
    "cudnnSetTensor4dDescriptor failed");
}

void Conv2dFwdGpuKernelMod::SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                           requested_algo_count, &returned_algo_count, &perf_results),
    "cudnnGetConvolutionForwardAlgorithm_v7 failed");
  conv_algorithm_ = perf_results.algo;
#if CUDNN_VERSION < 8000
  if (group_ > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardAlgorithm(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                          CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT, 0, &conv_algorithm_),
      "cudnnGetConvolutionForwardAlgorithm failed");
  }
#endif
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }
}

void Conv2dFwdGpuKernelMod::ResetResource() noexcept {
  conv_algorithm_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
  old_height_ = 0;
  old_width_ = 0;
  pad_height_ = 0;
  pad_width_ = 0;
  pad_top_ = 0;
  pad_left_ = 0;
  n_ = 0;
  c_ = 0;
  is_null_input_ = false;
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

void Conv2dFwdGpuKernelMod::InitSizeLists() {
  if (!is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(input_desc_, reinterpret_cast<size_t *>(&input_size_)),
      "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetFilterSizeInBytes(filter_desc_, reinterpret_cast<size_t *>(&filter_size_)),
      "cudnnGetFilterSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(output_desc_, reinterpret_cast<size_t *>(&output_size_)),
      "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(padded_desc_, reinterpret_cast<size_t *>(&padded_size_)),
      "cudnnGetTensorSizeInBytes failed");
  }
  input_size_list_.push_back(input_size_);
  input_size_list_.push_back(filter_size_);
  output_size_list_.push_back(output_size_);
  if (use_pad_ && !is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, padded_desc_, filter_desc_, conv_desc_, output_desc_,
                                              conv_algorithm_, &workspace_size_),
      "cudnnGetConvolutionForwardWorkspaceSize failed");
    workspace_size_list_.push_back(padded_size_);
  } else {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionForwardWorkspaceSize(cudnn_handle_, input_desc_, filter_desc_, conv_desc_, output_desc_,
                                                conv_algorithm_, &workspace_size_),
        "cudnnGetConvolutionForwardWorkspaceSize failed");
    }
  }
  (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);

  return;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv2D, Conv2dFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
