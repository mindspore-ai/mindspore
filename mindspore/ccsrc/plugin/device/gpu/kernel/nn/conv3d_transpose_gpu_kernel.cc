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

#include "plugin/device/gpu/kernel/nn/conv3d_transpose_gpu_kernel.h"
#include "mindspore/core/ops/conv3d_transpose.h"

namespace mindspore {
namespace kernel {
template <typename T>
bool Conv3dTransposeFwdGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                                  const std::vector<kernel::AddressPtr> &workspace,
                                                  const std::vector<kernel::AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *input_addr = GetDeviceAddress<T>(inputs, 0);
  T *filter_addr = GetDeviceAddress<T>(inputs, 1);
  T *output_addr = GetDeviceAddress<T>(outputs, 0);
  T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

  const float alpha = 1;
  if (use_pad_) {
    T *input_padded = GetDeviceAddress<T>(workspace, 1);
    const size_t kWsOutPadIdx = 2;
    T *output_padded = GetDeviceAddress<T>(workspace, kWsOutPadIdx);
    CalPad3d(input_padded_size_ / sizeof(T), input_addr, input_n_, input_c_, input_old_depth_, input_old_height_,
             input_old_width_, input_old_depth_ + pad_depth_, input_old_height_ + pad_height_,
             input_old_width_ + pad_width_, input_pad_head_, input_pad_top_, input_pad_left_, pad_value_, input_padded,
             reinterpret_cast<cudaStream_t>(cuda_stream_));
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_padded_descriptor_,
                                   input_padded, conv_desc_, algo_, work_space, workspace_size_, &beta_,
                                   padded_descriptor_, output_padded),
      "ConvolutionBackwardData failed");
    if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
      CalPadGrad3d(output_size_ / sizeof(T), output_padded, n_, c_, old_depth_, old_height_, old_width_,
                   old_depth_ + (1 + stride_[kDepth3DStrideIdx]) * pad_depth_,
                   old_height_ + (1 + stride_[kHeight3DStrideIdx]) * pad_height_,
                   old_width_ + (1 + stride_[kWidth3DStrideIdx]) * pad_width_, pad_head_, pad_top_, pad_left_,
                   output_addr, reinterpret_cast<cudaStream_t>(cuda_stream_));
    } else {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'data_format' only support 'NCDHW' right now "
                        << ", but got " << data_format_;
    }
  } else {
    if (greater_stride_) {
      T *stride_padded = GetDeviceAddress<T>(workspace, 1);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_desc_, input_addr,
                                     conv_desc_, algo_, work_space, workspace_size_, &beta_, stride_padded_descriptor_,
                                     stride_padded),
        "ConvolutionBackwardData failed");
      CalPad3d(output_size_ / sizeof(T), stride_padded, input_n_, input_c_, stride_pad_depth_, stride_pad_height_,
               stride_pad_width_, old_depth_, old_height_, old_width_, stride_pad_head_, stride_pad_top_,
               stride_pad_left_, pad_value_, output_addr, reinterpret_cast<cudaStream_t>(cuda_stream_));
    } else {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnConvolutionBackwardData(cudnn_handle_, &alpha, filter_desc_, filter_addr, input_desc_, input_addr,
                                     conv_desc_, algo_, work_space, workspace_size_, &beta_, output_desc_, output_addr),
        "ConvolutionBackwardData failed");
    }
  }
  return true;
}

using conv3dtransPair = std::pair<KernelAttr, Conv3dTransposeFwdGpuKernelMod::KernelRunFunc>;
const std::vector<conv3dtransPair> &Conv3dTransposeFwdGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, Conv3dTransposeFwdGpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &Conv3dTransposeFwdGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &Conv3dTransposeFwdGpuKernelMod::LaunchKernel<half>},
  };
  return func_list;
}

bool Conv3dTransposeFwdGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Conv3DTranspose>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  kernel_name_ = kernel_ptr->name();
  InitResource();

  size_t input_num = inputs.size();
  if (input_num != kInputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs must be 2, but got " << input_num;
    return false;
  }
  size_t output_num = outputs.size();
  if (output_num != kOutputNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs must be 1, but got " << output_num;
    return false;
  }

  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }

  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(kIndex0)->GetDtype()));
  data_format_ = kOpFormat_NCDHW;  // only support NCDHW right now
  format_attr_ = kernel_ptr->get_data_format();
  group_ = static_cast<int>(kernel_ptr->get_group());
  if (format_attr_ == kOpFormat_NDHWC) {
    data_format_ = kOpFormat_NDHWC;
  }
  return true;
}

int Conv3dTransposeFwdGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs,
                                           const std::map<uint32_t, tensor::TensorPtr> &) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::Conv3DTranspose>(base_operator);
  MS_ERROR_IF_NULL_W_RET_VAL(kernel_ptr, false);
  int ret = KRET_OK;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs)) != 0) {
    return ret;
  }

  auto filter_shape = inputs[kIndex1]->GetShapeVector();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  if (CheckNull(filter_shape, input_shape)) {
    return KRET_OK;
  }
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  if (output_shape.size() < kOutputShapeSize) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of output must be greater than or equal to 2, "
                  << "but got " << output_shape.size();
    return KRET_RESIZE_FAILED;
  }

  if (data_format_ == kOpFormat_NDHWC) {
    compute_format_ = CUDNN_TENSOR_NHWC;
    if (format_attr_ == kOpFormat_NCDHW) {
      ShapeNCDHW2NDHWC(&output_shape);
    }
  }
  SetNCDHW(output_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format_);
  SetNCDHW(input_shape, &input_n_, &input_c_, &input_old_depth_, &input_old_height_, &input_old_width_, data_format_);
  Set5DDesc(input_shape, output_shape, filter_shape);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                      "cudnnSetConvGroupCount failed");
  std::vector<int> pad_list;
  std::vector<int64_t> pad_list_me = GetValue<std::vector<int64_t>>(base_operator->GetAttr("pad_list"));
  (void)std::transform(pad_list_me.begin(), pad_list_me.end(), std::back_inserter(pad_list),
                       [](const int64_t &value) { return static_cast<int>(value); });
  std::vector<int> stride_pad_list(k3DPadSize, 0);
  std::vector<int64_t> stride_me = kernel_ptr->get_stride();
  std::vector<int64_t> dilation_me = kernel_ptr->get_dilation();
  SetStrideAndDilation(stride_me, dilation_me);
  pad_mode_ = kernel_ptr->get_pad_mode();
  SetPad(input_shape, filter_shape, &pad_list, &stride_pad_list);
  auto [input_desc_real, output_desc_real] = GetInputAndOutputDescReal(pad_list, stride_pad_list);
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                        "cudnnSetConvolutionMathType failed.")
  }
  SelectAlgorithm(input_desc_real, output_desc_real);

  if (base_operator->GetAttr("inplace_algo") == nullptr) {
    beta_ = 0;
  } else {
    beta_ = GetValue<std::string>(base_operator->GetAttr("inplace_algo")) == "cover" ? 0 : 1;
  }

  InitSizeLists();
  return KRET_OK;
}

void Conv3dTransposeFwdGpuKernelMod::SelectAlgorithm(cudnnTensorDescriptor_t input_desc_real,
                                                     cudnnTensorDescriptor_t output_desc_real) {
  constexpr int requested_algo_count = 1;
  constexpr int cudnn_major_num = 8;
  int returned_algo_count;
  cudnnConvolutionBwdDataAlgoPerf_t perf_results;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                                        cudnn_handle_, filter_desc_, input_desc_real, conv_desc_, output_desc_real,
                                        requested_algo_count, &returned_algo_count, &perf_results),
                                      "cudnnGetConvolutionBackwardDataAlgorithm_v7 failed");
  algo_ = perf_results.algo;
  if (compute_format_ == CUDNN_TENSOR_NHWC && cudnn_data_type_ == CUDNN_DATA_HALF && CUDNN_MAJOR < cudnn_major_num) {
    MS_LOG(ERROR) << "Conv3dTransposeFwdGpuKernelMod does not support float16 data with NDHWC format.";
  }
}

void Conv3dTransposeFwdGpuKernelMod::Set5DDesc(const ShapeVector &input_shape, const ShapeVector &output_shape,
                                               const ShapeVector &filter_shape) {
  const int kNbDims = 5;
  int dim_a[kNbDims];
  int stride_a_in[kNbDims];
  int dim_a_dy[kNbDims];
  int stride_a_dy[kNbDims];
  int filter_dim_a[kNbDims];
  SetDimA(output_shape, dim_a, kNbDims, data_format_);
  SetStrideA(output_shape, stride_a_in, kNbDims, data_format_);
  SetDimA(input_shape, dim_a_dy, kNbDims, data_format_);
  SetStrideA(input_shape, stride_a_dy, kNbDims, data_format_);
  SetDimA(filter_shape, filter_dim_a, kNbDims, data_format_);

  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(input_desc_, cudnn_data_type_, kNbDims, dim_a_dy, stride_a_dy),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(filter_desc_, cudnn_data_type_, compute_format_, kNbDims, filter_dim_a),
    "cudnnSetFilterNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(output_desc_, cudnn_data_type_, kNbDims, dim_a, stride_a_in),
    "cudnnSetTensorNdDescriptor failed");
}

void Conv3dTransposeFwdGpuKernelMod::SetStrideAndDilation(std::vector<int64_t> stride_me,
                                                          std::vector<int64_t> dilation_me) {
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

void Conv3dTransposeFwdGpuKernelMod::UpdatePaddingAndDilation(const ShapeVector &input_shape,
                                                              const ShapeVector &filter_shape, int *pad_list,
                                                              int *stride_pad_list) {
  const size_t kIdxOffset = 2;
  for (size_t i = 0; i < kConv3dDimSize; i++) {
    int pad_sum = LongToInt(filter_shape[i + kIdxOffset]) * dilation_[i + kIdxOffset] - stride_[i + kIdxOffset] -
                  dilation_[i + kIdxOffset] + 1;
    if (pad_sum >= 0) {
      int pad_0 = pad_sum / kSymmetricCoef;
      int pad_1 = pad_sum - pad_0;
      pad_list[i * kSymmetricCoef] = pad_0;
      pad_list[i * kSymmetricCoef + 1] = pad_1;
      stride_pad_list[i * kSymmetricCoef] = 0;
      stride_pad_list[i * kSymmetricCoef + 1] = 0;
    } else {  // pad_sum < 0, stride greater, need pad zero at end.
      pad_list[i * kSymmetricCoef] = 0;
      pad_list[i * kSymmetricCoef + 1] = 0;
      int pad_0 = (-pad_sum) / kSymmetricCoef;
      int pad_1 = (-pad_sum) - pad_0;
      stride_pad_list[i * kSymmetricCoef] = pad_0;
      stride_pad_list[i * kSymmetricCoef + 1] = pad_1;
      greater_stride_ = true;
    }
  }
}

void Conv3dTransposeFwdGpuKernelMod::UsePadProcess(const std::vector<int> &pad_list, const int *strideA,
                                                   const int *dilaA) {
  std::vector<int> padding_diff(kConv3dDimSize);
  std::vector<int> padding_common(kConv3dDimSize, 0);
  for (int i = 0; i < SizeToInt(kConv3dDimSize); i++) {
    padding_diff[i] = std::abs(pad_list[kSymmetricCoef * i + 1] - pad_list[kSymmetricCoef * i]);
    padding_common[i] = std::min(pad_list[kSymmetricCoef * i], pad_list[kSymmetricCoef * i + 1]);
  }
  pad_depth_ = padding_diff[kPadDepthIdx];
  pad_height_ = padding_diff[kPadHeightIdx];
  pad_width_ = padding_diff[kPadWidthIdx];
  pad_head_ = (pad_list[kHead3DPadIdx] - padding_common[kHead3DPadIdx]) * (stride_[kDepth3DStrideIdx] + 1);
  pad_top_ = (pad_list[kTop3DPadIdx] - padding_common[kTail3DPadIdx]) * (stride_[kHeight3DStrideIdx] + 1);
  pad_left_ = (pad_list[kLeft3DPadIdx] - padding_common[kTop3DPadIdx]) * (stride_[kWidth3DStrideIdx] + 1);
  input_pad_head_ = pad_list[kHead3DPadIdx] - padding_common[kHead3DPadIdx];
  input_pad_top_ = pad_list[kTop3DPadIdx] - padding_common[kTail3DPadIdx];
  input_pad_left_ = pad_list[kLeft3DPadIdx] - padding_common[kTop3DPadIdx];
  const size_t kDataSize = 5;
  int dim_a[kDataSize];
  int strideApadded[kDataSize];
  int input_dim_a[kDataSize];
  int input_strideApadded[kDataSize];
  if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
    ShapeVector padded_shape = {n_, c_, old_depth_ + (1 + stride_[kDepth3DStrideIdx]) * padding_diff[kHead3DPadIdx],
                                old_height_ + (1 + stride_[kHeight3DStrideIdx]) * padding_diff[kTail3DPadIdx],
                                old_width_ + (1 + stride_[kWidth3DStrideIdx]) * padding_diff[kTop3DPadIdx]};
    SetDimA(padded_shape, dim_a, kDataSize, data_format_);
    SetStrideA(padded_shape, strideApadded, kDataSize, data_format_);
    ShapeVector input_padded_shape = {input_n_, input_c_, input_old_depth_ + padding_diff[0],
                                      input_old_height_ + padding_diff[kTail3DPadIdx],
                                      input_old_width_ + padding_diff[kTop3DPadIdx]};
    SetDimA(input_padded_shape, input_dim_a, kDataSize, data_format_);
    SetStrideA(input_padded_shape, input_strideApadded, kDataSize, data_format_);
  } else if (data_format_ == kOpFormat_NDHWC) {
    ShapeVector padded_shape = {n_, old_depth_ + pad_depth_, old_height_ + pad_height_, old_width_ + pad_width_, c_};
    SetDimA(padded_shape, dim_a, kDataSize, data_format_);
    SetStrideA(padded_shape, strideApadded, kDataSize, data_format_);
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, kDataSize, dim_a, strideApadded),
    "cudnnSetTensor5dDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(input_padded_descriptor_, cudnn_data_type_, kDataSize, input_dim_a, input_strideApadded),
    "cudnnSetTensor5dDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetConvolutionNdDescriptor(conv_desc_, kConv3dDimSize, padding_common.data(), strideA, dilaA,
                                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT),
    "cudnnSetConvolutionNdDescriptor failed");
}

void Conv3dTransposeFwdGpuKernelMod::SetPad(const ShapeVector &input_shape, const ShapeVector &filter_shape,
                                            std::vector<int> *pad_list, std::vector<int> *stride_pad_list) {
  const size_t kFilterSize = 5;
  (void)CheckSize(filter_shape.size(), kFilterSize, "weight shape");
  (void)CheckSize(pad_list->size(), k3DPadSize, "pad");
  if (pad_mode_ == kSamePadModeUpperCase || pad_mode_ == kSamePadModeLowerCase) {  // pad_mode_ = same
    UpdatePaddingAndDilation(input_shape, filter_shape, pad_list->data(), stride_pad_list->data());
  }
  pad_depth_ = (*pad_list)[kHead3DPadIdx];
  pad_height_ = (*pad_list)[kTop3DPadIdx];
  pad_width_ = (*pad_list)[kLeft3DPadIdx];
  use_pad_ = !((pad_depth_ == (*pad_list)[kTail3DPadIdx]) && (pad_height_ == (*pad_list)[kBottom3DPadIdx]) &&
               (pad_width_ == (*pad_list)[kRight3DPadIdx]));
}

std::pair<cudnnTensorDescriptor_t, cudnnTensorDescriptor_t> Conv3dTransposeFwdGpuKernelMod::GetInputAndOutputDescReal(
  const std::vector<int> &pad_list, const std::vector<int> &stride_pad_list) {
  cudnnTensorDescriptor_t output_desc_real = nullptr;
  cudnnTensorDescriptor_t input_desc_real = nullptr;
  int strideA[kConv3dDimSize] = {stride_[kDepth3DStrideIdx], stride_[kHeight3DStrideIdx], stride_[kWidth3DStrideIdx]};
  int dilaA[kConv3dDimSize] = {dilation_[kDepth3DDilationIdx], dilation_[kHeight3DDilationIdx],
                               dilation_[kWidth3DDilationIdx]};
  if (use_pad_) {
    UsePadProcess(pad_list, strideA, dilaA);
    output_desc_real = padded_descriptor_;
    input_desc_real = input_padded_descriptor_;
  } else {
    if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
      pad_depth_ = 0;
      pad_height_ = 0;
      pad_width_ = 0;
    }
    int padA[kConv3dDimSize];
    padA[kPadDepthIdx] = pad_depth_;
    padA[kPadHeightIdx] = pad_height_;
    padA[kPadWidthIdx] = pad_width_;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv3dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolution3dDescriptor failed");
    if (greater_stride_) {
      stride_pad_head_ = stride_pad_list[kHead3DPadIdx];
      stride_pad_top_ = stride_pad_list[kTop3DPadIdx];
      stride_pad_left_ = stride_pad_list[kLeft3DPadIdx];
      stride_pad_depth_ = old_depth_ - stride_pad_list[kHead3DPadIdx] - stride_pad_list[kTail3DPadIdx];
      stride_pad_height_ = old_height_ - stride_pad_list[kTop3DPadIdx] - stride_pad_list[kBottom3DPadIdx];
      stride_pad_width_ = old_width_ - stride_pad_list[kLeft3DPadIdx] - stride_pad_list[kRight3DPadIdx];
      const size_t kDataLen = 5;
      int dim_a[kDataLen];
      int strideApadded[kDataLen];
      if (data_format_ == kOpFormat_NCDHW || data_format_ == kOpFormat_DEFAULT) {
        ShapeVector padded_shape = {n_, c_, stride_pad_depth_, stride_pad_height_, stride_pad_width_};
        SetDimA(padded_shape, dim_a, kDataLen, data_format_);
        SetStrideA(padded_shape, strideApadded, kDataLen, data_format_);
      }
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnSetTensorNdDescriptor(stride_padded_descriptor_, cudnn_data_type_, kDataLen, dim_a, strideApadded),
        "cudnnSetTensor5dDescriptor failed");
    }
    output_desc_real = greater_stride_ ? stride_padded_descriptor_ : output_desc_;
    input_desc_real = input_desc_;
  }

  return std::make_pair(input_desc_real, output_desc_real);
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv3DTranspose, Conv3dTransposeFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
