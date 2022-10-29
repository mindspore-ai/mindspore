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

#include "plugin/device/gpu/kernel/nn/conv2d_grad_filter_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/pad_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr int kInputDimSize = 4;
constexpr size_t kConv2dDimSize = 2;
constexpr int kSymmetricCoef = 2;

constexpr size_t k2DPadSize = 4;
constexpr size_t kTop2DPadIndex = 0;
constexpr size_t kBottom2DPadIndex = 1;
constexpr size_t kLeft2DPadIndex = 2;
constexpr size_t kRight2DPadIndex = 3;

constexpr size_t k2DStrideSize = 4;
constexpr size_t kHeight2DStrideIndex = 2;
constexpr size_t kWidth2DStrideIndex = 3;

constexpr size_t k2DDilationSize = 4;
constexpr size_t kHeight2DDilationIndex = 2;
constexpr size_t kWidth2DDilationIndex = 3;
constexpr auto StaticInput = 2;
constexpr auto DynamicInput = 3;

constexpr auto k2DHeightIndexNCHW = 2;
constexpr auto k2DHeightIndexNHWC = 1;

using KernelRunFunc = ConvGradFilterBkwGpuKernelMod::KernelRunFunc;
const std::vector<std::pair<KernelAttr, KernelRunFunc>> &ConvGradFilterBkwGpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list = {
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat32),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeFloat16),
     &ConvGradFilterBkwGpuKernelMod::LaunchKernel<half>}};
  return func_list;
}

bool ConvGradFilterBkwGpuKernelMod::Launch(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  stream_ptr_ = stream_ptr;
  return kernel_func_(this, inputs, workspace, outputs);
}

template <typename T>
bool ConvGradFilterBkwGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }
  T *dy = GetDeviceAddress<T>(inputs, 0);
  T *x = GetDeviceAddress<T>(inputs, 1);
  T *dw = GetDeviceAddress<T>(outputs, 0);
  T *work_space = GetPossiblyNullDeviceAddress<T>(workspace, 0);

  const float alpha = 1;
  const float beta = 0;

  if (use_pad_) {
    T *padded = GetDeviceAddress<T>(workspace, 1);
    if (data_format_ == kOpFormat_NHWC) {
      CalPadNHWC(padded_size_ / sizeof(T), x, n_, old_height_, old_width_, c_, old_height_ + pad_height_,
                 old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
                 reinterpret_cast<cudaStream_t>(stream_ptr_));
    } else {
      CalPad(padded_size_ / sizeof(T), x, n_, c_, old_height_, old_width_, old_height_ + pad_height_,
             old_width_ + pad_width_, pad_top_, pad_left_, pad_value_, padded,
             reinterpret_cast<cudaStream_t>(stream_ptr_));
    }
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, padded_descriptor_, padded, dy_desc_, dy, conv_desc_, algo_,
                                     work_space, workspace_size_, &beta, dw_desc_, dw),
      "ConvolutionBackwardFilter failed");
    return true;
  }
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnConvolutionBackwardFilter(cudnn_handle_, &alpha, x_desc_, x, dy_desc_, dy, conv_desc_, algo_, work_space,
                                   workspace_size_, &beta, dw_desc_, dw),
    "ConvolutionBackwardFilter failed");
  return true;
}

void ConvGradFilterBkwGpuKernelMod::CalPadList(const std::vector<int> &pad_list, const ShapeVector in_shape,
                                               const ShapeVector filter_shape, int h_index, int w_index) {
  if (pad_list[kTop2DPadIndex] == -1 || pad_list[kBottom2DPadIndex] == -1) {
    int pad_needed_h = (static_cast<int>(std::ceil((in_shape[h_index] * 1.0) / stride_[2])) - 1) * stride_[2] +
                       dilation_[2] * (filter_shape[h_index] - 1) + 1 - in_shape[h_index];
    pad_height_ = std::max(0, pad_needed_h);
    pad_top_ = static_cast<int>(std::floor(pad_height_ * 1.0 / kSymmetricCoef));
  } else {
    pad_height_ = pad_list[kTop2DPadIndex] + pad_list[kBottom2DPadIndex];
    pad_top_ = pad_list[kTop2DPadIndex];
  }
  if (pad_list[kLeft2DPadIndex] == -1 || pad_list[kRight2DPadIndex] == -1) {
    int pad_needed_w = (static_cast<int>(std::ceil((in_shape[w_index] * 1.0) / stride_[3])) - 1) * stride_[3] +
                       dilation_[3] * (filter_shape[w_index] - 1) + 1 - in_shape[w_index];
    pad_width_ = std::max(0, pad_needed_w);
    pad_left_ = static_cast<int>(std::floor(pad_width_ * 1.0 / kSymmetricCoef));
  } else {
    pad_width_ = pad_list[kLeft2DPadIndex] + pad_list[kRight2DPadIndex];
    pad_left_ = pad_list[kLeft2DPadIndex];
  }
}

void ConvGradFilterBkwGpuKernelMod::CheckParam(const std::vector<KernelTensorPtr> &inputs) {
  size_t input_num = inputs.size();
  if (input_num != StaticInput && input_num != DynamicInput) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the number of inputs must be 2 or 3, but got " << input_num;
  }
  if (input_num == DynamicInput) {
    is_dynamic_attr_ = true;
  }
}

void ConvGradFilterBkwGpuKernelMod::Set4DDesc(const ShapeVector &dy_shape, const ShapeVector &filter_shape,
                                              const ShapeVector &in_shape) {
  int dimA[kInputDimSize];
  int strideAin[kInputDimSize];
  int dimAdy[kInputDimSize];
  int strideAdy[kInputDimSize];
  SetDimA(in_shape, dimA, kInputDimSize, data_format_);
  SetStrideA(in_shape, strideAin, kInputDimSize, data_format_);
  SetDimA(dy_shape, dimAdy, kInputDimSize, data_format_);
  SetStrideA(dy_shape, strideAdy, kInputDimSize, data_format_);
  // filter shape relued by format_attr_. In native mode it's OHWI. In transpose mode it's OIHW.
  int filterDimA[kInputDimSize];
  SetDimA(filter_shape, filterDimA, kInputDimSize, format_attr_);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(dy_desc_, cudnn_data_type_, kInputDimSize, dimAdy, strideAdy),
    "cudnnSetTensorNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetFilterNdDescriptor(dw_desc_, cudnn_data_type_, compute_format_, kInputDimSize, filterDimA),
    "cudnnSetFilterNdDescriptor failed");
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(x_desc_, cudnn_data_type_, kInputDimSize, dimA, strideAin),
    "cudnnSetTensorNdDescriptor failed");
}

void ConvGradFilterBkwGpuKernelMod::SetStrideAndDilation(const std::vector<int64_t> &stride_me,
                                                         const std::vector<int64_t> &dilation_me) {
  (void)std::transform(stride_me.begin(), stride_me.end(), std::back_inserter(stride_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(dilation_me.begin(), dilation_me.end(), std::back_inserter(dilation_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (stride_.size() != k2DStrideSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'stride' must be 4, but got " << stride_.size();
  }
  if (stride_[0] != 1 || stride_[1] != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'stride' at 0 and 1 axis must be 1, but got "
                      << "stride[0]: " << stride_[0] << ", stride[1]: " << stride_[1];
  }
  if (dilation_.size() != k2DDilationSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'dilation' must be 4, but got "
                      << dilation_.size();
  }
  if (dilation_[0] != 1 || dilation_[1] != 1) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the value of 'dilation' at 0 and 1 axis must be 1, but got "
                      << "dilation[0]: " << dilation_[0] << ", dilation[1]: " << dilation_[1];
  }
}

void ConvGradFilterBkwGpuKernelMod::SelectAlgorithm(cudnnTensorDescriptor_t x_desc_real) {
  constexpr int requested_algo_count = 1;
  int returned_algo_count = 0;
  cudnnConvolutionBwdFilterAlgoPerf_t perf_results;
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnGetConvolutionBackwardFilterAlgorithm_v7(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                  requested_algo_count, &returned_algo_count, &perf_results),
    "GetConvolutionBackwardFilterAlgorithm failed");
  algo_ = perf_results.algo;
#if CUDNN_VERSION < 8000
  if (group_ > 1) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardFilterAlgorithm(cudnn_handle_, x_desc_real, dy_desc_, conv_desc_, dw_desc_,
                                                 CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT, 0, &algo_),
      "GetConvolutionBackwardFilterAlgorithm failed");
  }
#endif
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_1;
  }
}

void ConvGradFilterBkwGpuKernelMod::InitSizeLists() {
  if (!is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dy_desc_, reinterpret_cast<size_t *>(&dy_size_)),
                                        "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(x_desc_, reinterpret_cast<size_t *>(&input_size_)),
                                        "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetFilterSizeInBytes(dw_desc_, reinterpret_cast<size_t *>(&output_size_)),
                                        "cudnnGetFilterSizeInBytes failed");
  }
  input_size_list_.push_back(dy_size_);
  input_size_list_.push_back(input_size_);
  output_size_list_.push_back(output_size_);

  if (use_pad_ && !is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetTensorSizeInBytes(padded_descriptor_, reinterpret_cast<size_t *>(&padded_size_)),
      "cudnnGetTensorSizeInBytes failed");
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, padded_descriptor_, dy_desc_, conv_desc_, dw_desc_,
                                                     algo_, reinterpret_cast<size_t *>(&workspace_size_)),
      "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
    workspace_size_list_.push_back(padded_size_);
  } else {
    if (!is_null_input_) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnGetConvolutionBackwardFilterWorkspaceSize(cudnn_handle_, x_desc_, dy_desc_, conv_desc_, dw_desc_, algo_,
                                                       reinterpret_cast<size_t *>(&workspace_size_)),
        "cudnnGetConvolutionBackwardFilterWorkspaceSize failed");
    }
  }
  (void)workspace_size_list_.insert(workspace_size_list_.begin(), workspace_size_);
}

bool ConvGradFilterBkwGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  if (!MatchKernelFunc(base_operator, inputs, outputs)) {
    return false;
  }
  kernel_name_ = base_operator->name();
  InitResource();
  CheckParam(inputs);
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs[0]->GetDtype()));
  data_format_ = mindspore::FormatEnumToString(inputs[0]->GetFormat());
  auto prim = base_operator->GetPrim();
  MS_EXCEPTION_IF_NULL(prim);
  format_attr_ = GetValue<std::string>(prim->GetAttr("format"));
  if (format_attr_ == kOpFormat_NHWC) {
    data_format_ = kOpFormat_NHWC;
  }
  if (!is_dynamic_attr_) {
    filter_shape_ = GetValue<std::vector<int64_t>>(prim->GetAttr("filter_sizes"));
  }
  group_ = static_cast<int>(GetValue<int64_t>(prim->GetAttr("group")));
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionGroupCount(conv_desc_, group_),
                                      "cudnnSetConvGroupCount failed");
  std::vector<int64_t> pad_list_ori = GetValue<std::vector<int64_t>>(base_operator->GetAttr("pad_list"));
  (void)std::transform(pad_list_ori.begin(), pad_list_ori.end(), std::back_inserter(pad_list_),
                       [](const int64_t &value) { return static_cast<int>(value); });
  if (pad_list_.size() != k2DPadSize) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad' must be 4, but got " << pad_list_.size();
  }
  auto stride_ori = GetValue<std::vector<int64_t>>(prim->GetAttr("stride"));
  auto dilation_ori = GetValue<std::vector<int64_t>>(prim->GetAttr("dilation"));
  SetStrideAndDilation(stride_ori, dilation_ori);
  pad_mode_ = GetValue<std::string>(prim->GetAttr("pad_mode"));
  return true;
}

int ConvGradFilterBkwGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost); ret != KRET_OK) {
    return ret;
  }
  ResetResource();
  auto dy_shape = inputs[0]->GetDeviceShapeAdaptively();
  auto in_shape = inputs[1]->GetDeviceShapeAdaptively();
  is_null_input_ = CHECK_SHAPE_NULL(dy_shape, kernel_name_, "dy") || CHECK_SHAPE_NULL(in_shape, kernel_name_, "x");
  if (is_null_input_) {
    return KRET_OK;
  }
  if (is_dynamic_attr_) {
    constexpr size_t kShapeIndex = 2;
    auto value_res = TryGetIntValueFromInputs(inputs, kShapeIndex, kernel_name_, true);
    if (!value_res.has_value()) {
      MS_LOG(EXCEPTION) << "Fail to get filter_sizes from inputs";
    }
    filter_shape_ = value_res.value();
  }
  auto filter_shape = filter_shape_;
  CheckTensorSize({in_shape, dy_shape, filter_shape});
  int h_index = k2DHeightIndexNCHW;
  int w_index = k2DHeightIndexNCHW + 1;
  if (data_format_ == kOpFormat_NHWC) {
    compute_format_ = CUDNN_TENSOR_NHWC;
    h_index = k2DHeightIndexNHWC;
    w_index = k2DHeightIndexNHWC + 1;
  }
  SetNCHW(in_shape, &n_, &c_, &old_height_, &old_width_, data_format_);
  Set4DDesc(dy_shape, filter_shape, in_shape);
  CalPadList(pad_list_, in_shape, filter_shape, h_index, w_index);
  use_pad_ = !(pad_height_ % kSymmetricCoef == 0 && pad_width_ % kSymmetricCoef == 0);
  cudnnTensorDescriptor_t x_desc_real = nullptr;
  int padA[kConv2dDimSize];
  int strideA[kConv2dDimSize] = {stride_[kHeight2DStrideIndex], stride_[kWidth2DStrideIndex]};
  int dilaA[kConv2dDimSize] = {dilation_[kHeight2DDilationIndex], dilation_[kWidth2DDilationIndex]};
  if (use_pad_) {
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
      cudnnSetTensorNdDescriptor(padded_descriptor_, cudnn_data_type_, kInputDimSize, dimA, strideApadded),
      "cudnnSetTensor4dDescriptor failed");
    padA[0] = 0;
    padA[1] = 0;
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolutionNdDescriptor failed");
    x_desc_real = padded_descriptor_;
  } else {
    if (pad_mode_ == kValidPadModeUpperCase || pad_mode_ == kValidPadModeLowerCase) {
      pad_top_ = 0;
      pad_left_ = 0;
    }
    padA[0] = pad_top_;
    padA[1] = pad_left_;

    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnSetConvolutionNdDescriptor(conv_desc_, kConv2dDimSize, padA, strideA, dilaA, CUDNN_CROSS_CORRELATION,
                                      CUDNN_DATA_FLOAT),
      "cudnnSetConvolution2dDescriptor failed");
    x_desc_real = x_desc_;
  }
  if (cudnn_data_type_ == CUDNN_DATA_HALF) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH),
                                        "cudnnSetConvolutionMathType failed.")
  }
  SelectAlgorithm(x_desc_real);
  InitSizeLists();
  return KRET_OK;
}

void ConvGradFilterBkwGpuKernelMod::ResetResource() noexcept {
  algo_ = CUDNN_CONVOLUTION_BWD_FILTER_ALGO_0;
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
  dy_size_ = 0;
  output_size_ = 0;
  padded_size_ = 0;
  workspace_size_ = 0;
  use_pad_ = True;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Conv2DBackpropFilter, ConvGradFilterBkwGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
