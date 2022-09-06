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

#include "plugin/device/gpu/kernel/nn/conv3d_gpu_kernel.h"
#include "mindspore/core/ops/conv3d.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool Conv3dGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
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
    CalPad3d(padded_size_ / sizeof(T), input_addr, n_, c_, old_depth_, old_height_, old_width_, old_depth_ + pad_depth_,
             old_height_ + pad_height_, old_width_ + pad_width_, pad_head_, pad_top_, pad_left_, pad_value_,
             padded_addr, reinterpret_cast<cudaStream_t>(cuda_stream_));
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

void Conv3dGpuKernelMod::CheckSize(const size_t value, const size_t expect_value, const string arg_name) {
  if (value != expect_value) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dimension of " << arg_name << " must be " << expect_value
                      << ", but got " << value;
  }
}

using conv3dPair = std::pair<KernelAttr, Conv3dGpuKernelMod::KernelRunFunc>;
const std::vector<conv3dPair> &Conv3dGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, Conv3dGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &Conv3dGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &Conv3dGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}

bool Conv3dGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                              const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Conv3D>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();
  InitResource();
  size_t input_num = inputs.size();
  const size_t kInputNum = 2;
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return false;
  }
  size_t output_num = outputs.size();
  if (output_num != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));
  data_format_ = kOpFormat_NCDHW;
  return true;
}

int Conv3dGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs,
                               const std::map<uint32_t, tensor::TensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Conv3D>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }
  auto in_shape = inputs[kIndex0]->GetShapeVector();
  auto filter_shape = inputs[kIndex1]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  is_null_input_ = CHECK_SHAPE_NULL(in_shape, kernel_name_, "x") ||
                   CHECK_SHAPE_NULL(filter_shape, kernel_name_, "weight") ||
                   CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    InitSizeLists();
    return KRET_OK;
  }
  CheckTensorSize({in_shape});
  (void)CheckSize(in_shape.size(), kInputDimSize, "x");
  n_ = LongToInt(in_shape[kInDimIdxForN]);
  c_ = LongToInt(in_shape[kInDimIdxForC]);
  old_depth_ = LongToInt(in_shape[kInDimIdxForD]);
  old_height_ = LongToInt(in_shape[kInDimIdxForH]);
  old_width_ = LongToInt(in_shape[kInDimIdxForW]);
  compute_format_ = CUDNN_TENSOR_NCHW;
  SetNDDesc(in_shape, filter_shape, output_shape);
  group_ = static_cast<int>(kernel_ptr->get_group());
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                      "cudnnSetConvGroupCount failed");
  std::vector<int> pad_list;
  std::vector<int64_t> pad_list_me = GetValue<std::vector<int64_t>>(base_operator->GetAttr("pad_list"));
  (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (pad_list.size() != k3DPadSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the length of 'pad' must be 6, but got " << pad_list.size();
    return KRET_RESIZE_FAILED;
  }
  pad_mode_ = kernel_ptr->get_pad_mode();
  SetPad(pad_list);
  std::vector<int64_t> stride_me = kernel_ptr->get_stride();
  std::vector<int64_t> dilation_me = kernel_ptr->get_dilation();
  SetStrideAndDilation(stride_me, dilation_me);
  auto input_descriptor_real = GetInputDescReal(pad_list);
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                        "cudnnSetConvolutionMathType failed.")
  }
  SelectAlgorithm(input_descriptor_real);
  InitSizeLists();
  return KRET_OK;
}

void Conv3dGpuKernelMod::SetNDDesc(const ShapeVector &in_shape, const ShapeVector &filter_shape,
                                   const ShapeVector &output_shape) {
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
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, kDims, dimA, strideAin),
                                      "cudnnSetTensor4dDescriptor failed");

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, kDims, filterDimA),
    "cudnnSetFilter4dDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, kDims, dimAout, strideAout),
    "cudnnSetTensor4dDescriptor failed");
}

void Conv3dGpuKernelMod::SelectAlgorithm(cudnnTensorDescriptor_t input_descriptor_real) {
  const int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionFwdAlgoPerf_t perf_results;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetConvolutionForwardAlgorithm_v7(cudnn_handle_, input_descriptor_real, filter_desc_, conv_desc_, output_desc_,
                                           requested_algo_count, &returned_algo_count, &perf_results),
    "cudnnGetConvolutionForwardAlgorithm_v7 failed");
  conv_algorithm_ = perf_results.algo;
}

void Conv3dGpuKernelMod::SetStrideAndDilation(std::vector<int64_t> stride_me, std::vector<int64_t> dilation_me) {
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

void Conv3dGpuKernelMod::SetPad(const std::vector<int> &pad_list) {
  pad_depth_ = pad_list[kHead3DPadIdx];
  pad_height_ = pad_list[kTop3DPadIdx];
  pad_width_ = pad_list[kLeft3DPadIdx];
  use_pad_ = (pad_depth_ != pad_list[kTail3DPadIdx]) || (pad_height_ != pad_list[kBottom3DPadIdx]) ||
             (pad_width_ != pad_list[kRight3DPadIdx]);
}

cudnnTensorDescriptor_t Conv3dGpuKernelMod::GetInputDescReal(const std::vector<int> &pad_list) {
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
    ShapeVector padded_shape = {n_, c_, old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_};
    SetDimA(padded_shape, dimA, kNumDims, data_format_);
    SetStrideA(padded_shape, strideApadded, kNumDims, data_format_);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetTensorNdDescriptor(padded_desc_, cudnn_data_type_, kNumDims, dimA, strideApadded),
      "cudnnSetTensor4dDescriptor failed");
    padA[kPadDepthIdx] = 0;
    padA[kPadHeightIdx] = 0;
    padA[kPadWidthIdx] = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
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
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionNdDescriptor(conv_desc_, kConvDims, padA, strideA, dilaA,
                                                                        CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
                                        "cudnnSetConvolution2dDescriptor failed");
    input_descriptor_real = input_desc_;
  }
  return input_descriptor_real;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv3D, Conv3dGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
