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

#include "plugin/device/gpu/kernel/nn/pooling_grad_gpu_kernel.h"
#include <functional>
#include <memory>
#include "mindspore/core/ops/grad/pool_grad.h"
#include "mindspore/core/ops/grad/avg_pool_3d_grad.h"
#include "mindspore/core/ops/grad/max_pool_3d_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/broadcast_impl.cuh"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/avg_pool3d_helper_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr auto kMaxPoolGrad = "MaxPoolGrad";
constexpr auto kMaxPool3DGrad = "MaxPool3DGrad";
constexpr auto kAvgPoolGrad = "AvgPoolGrad";
constexpr auto kAvgPool3DGrad = "AvgPool3DGrad";
constexpr size_t kInputNum = 3;
constexpr size_t kAvgPool3DGradDynamicInputNum = 2;

bool PoolingGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto pool_grad_ptr = std::make_shared<ops::PoolGrad>(base_operator->GetPrim());
  format_attr_ = pool_grad_ptr->get_format();
  pad_mode_ = pool_grad_ptr->get_pad_mode();
  stride_me_ = pool_grad_ptr->get_strides();
  window_me_ = pool_grad_ptr->get_kernel_size();
  if (kernel_name_ == kMaxPool3DGrad) {
    auto kernel_ptr = std::make_shared<ops::MaxPool3DGrad>(base_operator->GetPrim());
    pad_list_ = kernel_ptr->get_pad_list();
  } else if (kernel_name_ == kAvgPool3DGrad) {
    auto kernel_ptr = std::make_shared<ops::AvgPool3DGrad>(base_operator->GetPrim());
    pad_list_ = kernel_ptr->get_pad_list();
    divisor_override_ = kernel_ptr->get_divisor_override();
    ceil_mode_ = kernel_ptr->get_ceil_mode();
    include_ = kernel_ptr->get_count_include_pad();
  }
  SetFirstInputIndex(inputs.size());
  cudnn_data_type_ = GetCudnnDataType(TypeIdLabel(inputs.at(first_input_index_)->GetDtype()));
  SetPoolingMode();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = kernel_attr_map_.at(kernel_name_)[index].second;
  return true;
}

inline void CheckInputNum(const std::string &kernel_name, const size_t input_num) {
  if (kernel_name != kAvgPool3DGrad) {
    if (input_num != kInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs must be " << kInputNum << ", but got "
                        << input_num;
    }
  } else {
    if (input_num != kAvgPool3DGradDynamicInputNum) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name << "', the number of inputs must be " << kAvgPool3DGradDynamicInputNum
                        << ", but got " << input_num;
    }
  }
}

int PoolingGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  CheckInputNum(kernel_name_, inputs.size());
  input_shape_ = inputs.at(first_input_index_)->GetShapeVector();
  output_shape_ = outputs.at(kIndex0)->GetShapeVector();
  int nbDims = SizeToInt(input_shape_.size());
  int dimA[kPoolingNbDims];
  int strideAin[kPoolingNbDims];
  int dimAy[kPoolingNbDims];
  int strideAiny[kPoolingNbDims];
  int dimAdy[kPoolingNbDims];
  int strideAdy[kPoolingNbDims];
  int dimAout[kPoolingNbDims];
  int strideAout[kPoolingNbDims];
  if (!InitShape(inputs, outputs, dimA, strideAin, dimAy, strideAiny, dimAdy, strideAdy, dimAout, strideAout, nbDims)) {
    return ret;
  }
  if (nbDims == kDim2DShapeSize) {
    SetPad();
  } else if (nbDims == kDim3DShapeSize) {
    SetPad3D();
  }
  std::string err_msg = "For '" + kernel_name_ + "', cudnnSetTensor4dDescriptor failed";
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(y_descriptor_, cudnn_data_type_, nbDims, dimAy, strideAiny), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(dy_descriptor_, cudnn_data_type_, nbDims, dimAdy, strideAdy), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(dx_descriptor_, cudnn_data_type_, nbDims, dimAout, strideAout), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetTensorNdDescriptor(x_descriptor_, cudnn_data_type_, nbDims, dimA, strideAin), err_msg);
  edge_kernel_ = GetEdgeKernelSize();
  InitSizeLists();
  return ret;
}

template <typename T>
bool PoolingGradGpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                           const std::vector<kernel::AddressPtr> &workspace,
                                           const std::vector<kernel::AddressPtr> &outputs) {
  T *x_data = nullptr;
  T *y = nullptr;
  T *dy = nullptr;
  T *dx = nullptr;
  if (kernel_name_ == kAvgPool3DGrad) {
    dy = GetDeviceAddress<T>(inputs, first_input_index_);
    dx = GetDeviceAddress<T>(outputs, kIndex0);
    x_data = GetDeviceAddress<T>(workspace, kIndex0);
    y = GetDeviceAddress<T>(workspace, kIndex1);
  } else {
    x_data = GetDeviceAddress<T>(inputs, kIndex0);
    y = GetDeviceAddress<T>(inputs, kIndex1);
    dy = GetDeviceAddress<T>(inputs, kIndex2);
    dx = GetDeviceAddress<T>(outputs, kIndex0);
  }
  T alpha = static_cast<T>(1.0f);
  T beta = static_cast<T>(0.0f);
  if (divisor_override_ != 0) {
    T *work_addr = GetDeviceAddress<T>(workspace, kIndex2);
    T *dy_work_addr = GetDeviceAddress<T>(workspace, kIndex3);
    size_t output_num = input_size_ / sizeof(T);
    int64_t size = std::accumulate(kernel_size_.begin(), kernel_size_.end(), int64_t(1), std::multiplies<int64_t>());
    T divisor = static_cast<T>(LongToFloat(size) / LongToFloat(divisor_override_));
    std::vector<T> divisor_value(output_num, divisor);
    std::string err_msg = "For '" + kernel_name_ + "', cudaMemcpyAsync failed.";
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(work_addr, divisor_value.data(), input_size_, cudaMemcpyHostToDevice,
                      reinterpret_cast<cudaStream_t>(cuda_stream_)),
      err_msg);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaMemcpyAsync(dy_work_addr, dy, input_size_, cudaMemcpyDeviceToDevice,
                                                       reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       err_msg);
    if (ceil_mode_) {
      CalRealKernelSize(input_shape_, kernel_size_, edge_kernel_, work_addr, device_id_,
                        reinterpret_cast<cudaStream_t>(cuda_stream_));
    }
    ElewiseArith(output_num, BinaryOpType::kMul, dy_work_addr, work_addr, dy_work_addr,
                 reinterpret_cast<cudaStream_t>(cuda_stream_));
    if (cudnn_data_type_ == CUDNN_DATA_DOUBLE) {
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy_work_addr,
                             x_descriptor_, x_data, &beta, dx_descriptor_, dx),
        "For '" + kernel_name_ + "', cudnnPoolingBackward failed");
    } else {
      const float alphaf = static_cast<float>(alpha);
      const float betaf = static_cast<float>(beta);
      CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
        cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alphaf, y_descriptor_, y, dy_descriptor_,
                             dy_work_addr, x_descriptor_, x_data, &betaf, dx_descriptor_, dx),
        "For '" + kernel_name_ + "', cudnnPoolingBackward failed");
    }

    return true;
  }
  if (cudnn_data_type_ == CUDNN_DATA_DOUBLE) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alpha, y_descriptor_, y, dy_descriptor_, dy,
                           x_descriptor_, x_data, &beta, dx_descriptor_, dx),
      "For '" + kernel_name_ + "', cudnnPoolingBackward failed");
  } else {
    const float alphaf = static_cast<float>(alpha);
    const float betaf = static_cast<float>(beta);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
      cudnnPoolingBackward(cudnn_handle_, pooling_descriptor_, &alphaf, y_descriptor_, y, dy_descriptor_, dy,
                           x_descriptor_, x_data, &betaf, dx_descriptor_, dx),
      "For '" + kernel_name_ + "', cudnnPoolingBackward failed");
  }

  return true;
}

bool PoolingGradGpuKernelMod::InitShape(const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs, int *dimA, int *strideAin,
                                        int *dimAy, int *strideAiny, int *dimAdy, int *strideAdy, int *dimAout,
                                        int *strideAout, int nbDims) {
  ShapeVector dout_shape, input_mask, output_shape, input_shape;
  if (kernel_name_ == kAvgPool3DGrad) {
    dout_shape = inputs.at(first_input_index_)->GetDeviceShapeAdaptively();
    output_shape = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
    input_mask = dout_shape;
    input_shape = output_shape;
  } else {
    input_shape = inputs.at(kIndex0)->GetDeviceShapeAdaptively();
    input_mask = inputs.at(kIndex1)->GetDeviceShapeAdaptively();
    dout_shape = inputs.at(kIndex2)->GetDeviceShapeAdaptively();
    output_shape = outputs.at(kIndex0)->GetDeviceShapeAdaptively();
  }
  is_null_input_ =
    CHECK_SHAPE_NULL(input_shape, kernel_name_, "input") || CHECK_SHAPE_NULL(input_mask, kernel_name_, "mask") ||
    CHECK_SHAPE_NULL(dout_shape, kernel_name_, "dout") || CHECK_SHAPE_NULL(output_shape, kernel_name_, "output");
  if (is_null_input_) {
    InitSizeLists();
    return false;
  }
  auto data_format = GetFormatFromEnumToStr(inputs.at(first_input_index_)->GetFormat());
  if (Anyone(format_attr_, Format::NHWC, Format::NDHWC)) {
    data_format = GetFormatFromEnumToStr(format_attr_);
  }

  CheckTensorSize({input_shape, input_mask, dout_shape, output_shape});
  if (nbDims == kDim2DShapeSize) {
    SetNCHW(input_shape, &n_, &c_, &old_height_, &old_width_, data_format);
  } else if (nbDims == kDim3DShapeSize) {
    SetNCDHW(input_shape, &n_, &c_, &old_depth_, &old_height_, &old_width_, data_format);
  }
  SetDimA(input_shape, dimA, nbDims, data_format);
  SetStrideA(input_shape, strideAin, nbDims, data_format);
  SetDimA(input_mask, dimAy, nbDims, data_format);
  SetStrideA(input_mask, strideAiny, nbDims, data_format);
  SetDimA(dout_shape, dimAdy, nbDims, data_format);
  SetStrideA(dout_shape, strideAdy, nbDims, data_format);
  SetDimA(output_shape, dimAout, nbDims, data_format);
  SetStrideA(output_shape, strideAout, nbDims, data_format);
  return true;
}

void PoolingGradGpuKernelMod::DestroyResource() noexcept {
  std::string err_msg = "For '" + kernel_name_ + "', cudnnDestroyPoolingDescriptor failed";
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyPoolingDescriptor(pooling_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(dx_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(x_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(dy_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnDestroyTensorDescriptor(y_descriptor_), err_msg);
}

void PoolingGradGpuKernelMod::InitResource() {
  pooling_mode_ = CUDNN_POOLING_MAX;
  cudnn_data_type_ = CUDNN_DATA_FLOAT;
  compute_format_ = CUDNN_TENSOR_NCHW;
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  std::string err_msg = "For '" + kernel_name_ + "', cudnnCreateTensorDescriptor failed";
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&y_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dy_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&x_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreateTensorDescriptor(&dx_descriptor_), err_msg);
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnCreatePoolingDescriptor(&pooling_descriptor_), err_msg);
}

void PoolingGradGpuKernelMod::InitSizeLists() {
  if (first_input_index_ == 1) {
    input_size_list_.resize(1);
  } else {
    input_size_list_.clear();
  }
  output_size_list_.clear();
  workspace_size_list_.clear();
  std::string err_msg = "For '" + kernel_name_ + "', cudnnGetTensorSizeInBytes failed";
  if (!is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(x_descriptor_, &input_size_), err_msg);
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dx_descriptor_, &output_size_), err_msg);
  }
  input_size_list_.push_back(input_size_);
  output_size_list_.push_back(output_size_);
  if (kernel_name_ == kAvgPool3DGrad) {
    workspace_size_list_.push_back(output_size_);
    workspace_size_list_.push_back(input_size_);
    if (divisor_override_ != 0) {
      workspace_size_list_.push_back(input_size_);
      workspace_size_list_.push_back(input_size_);
    }
  }
  if (!is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(y_descriptor_, &input_size_), err_msg);
  }
  input_size_list_.push_back(input_size_);

  if (!is_null_input_) {
    CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(cudnnGetTensorSizeInBytes(dy_descriptor_, &input_size_), err_msg);
  }
  input_size_list_.push_back(input_size_);
}

void PoolingGradGpuKernelMod::SetPad() {
  std::vector<int> window;
  std::vector<int> stride;
  (void)std::transform(stride_me_.begin(), stride_me_.end(), std::back_inserter(stride),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(window_me_.begin(), window_me_.end(), std::back_inserter(window),
                       [](const int64_t &value) { return static_cast<int>(value); });
  const size_t kSizeLowerLimit = 4;
  const size_t kIdxH = 2;
  const size_t kIdxW = 3;
  if (window.size() < kSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 4, but got "
                      << window.size();
  }
  int window_height = window[kIdxH];
  int window_width = window[kIdxW];
  if (stride.size() < kSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 4, but got "
                      << stride.size();
  }
  int stride_h = stride[kIdxH];
  int stride_w = stride[kIdxW];
  if (format_attr_ == Format::NHWC) {
    const size_t kNHWCIdxH = 1;
    const size_t kNHWCIdxW = 2;
    window_height = window[kNHWCIdxH];
    window_width = window[kNHWCIdxW];
    stride_h = stride[kNHWCIdxH];
    stride_w = stride[kNHWCIdxW];
  }
  const size_t k2dDim = 2;
  int windowDimA[k2dDim] = {window_height, window_width};
  int paddingA[k2dDim] = {0, 0};
  int strideA[k2dDim] = {stride_h, stride_w};
  if (pad_mode_ == PadMode::SAME) {
    pad_height_ = GetPad(old_height_, window_height, stride_h);
    pad_width_ = GetPad(old_width_, window_width, stride_w);
    const int kSymCoef = 2;
    pad_top_ = pad_height_ / kSymCoef;
    pad_left_ = pad_width_ / kSymCoef;
    paddingA[kIndex0] = pad_top_;
    paddingA[kIndex1] = pad_left_;
  } else {
    if (pad_mode_ == PadMode::VALID) {
      pad_height_ = 0;
      pad_width_ = 0;
    }
  }
  std::string err_msg = "For '" + kernel_name_ + "', cudnnSetPoolingNdDescriptor failed";
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, k2dDim, windowDimA,
                                paddingA, strideA),
    err_msg);
}

void PoolingGradGpuKernelMod::SetPad3D() {
  const int kPadListSize = 6;
  const int kPadScale = 2;
  std::vector<int> window;
  std::vector<int> stride;
  (void)std::transform(stride_me_.begin(), stride_me_.end(), std::back_inserter(stride),
                       [](const int64_t &value) { return static_cast<int>(value); });
  (void)std::transform(window_me_.begin(), window_me_.end(), std::back_inserter(window),
                       [](const int64_t &value) { return static_cast<int>(value); });
  const size_t k3dSizeLowerLimit = 5;
  const size_t kIdxD = 2;
  const size_t kIdxH = 3;
  const size_t kIdxW = 4;
  if (window.size() < k3dSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'kernel_size' cannot be less than 5, but got "
                      << window.size();
  }
  int window_depth = window[kIdxD];
  int window_height = window[kIdxH];
  int window_width = window[kIdxW];
  if (stride.size() < k3dSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'strides' cannot be less than 5, but got "
                      << stride.size();
  }
  int stride_d = stride[kIdxD];
  int stride_h = stride[kIdxH];
  int stride_w = stride[kIdxW];
  if (format_attr_ == Format::NDHWC) {
    const size_t kNDHWCIdxD = 1;
    const size_t kNDHWCIdxH = 2;
    const size_t kNDHWCIdxW = 3;
    window_depth = window[kNDHWCIdxD];
    window_height = window[kNDHWCIdxH];
    window_width = window[kNDHWCIdxW];
    stride_d = stride[kNDHWCIdxD];
    stride_h = stride[kNDHWCIdxH];
    stride_w = stride[kNDHWCIdxW];
  }
  const size_t k3dDimSize = 3;
  int windowDimA[k3dDimSize] = {window_depth, window_height, window_width};
  int paddingA[k3dDimSize] = {0, 0, 0};
  int strideA[k3dDimSize] = {stride_d, stride_h, stride_w};
  if (pad_mode_ == PadMode::SAME) {
    pad_depth_ = GetPad(old_depth_, window_depth, stride_d);
    pad_height_ = GetPad(old_height_, window_height, stride_h);
    pad_width_ = GetPad(old_width_, window_width, stride_w);
    const int kSymCoef = 2;
    pad_front_ = pad_depth_ / kSymCoef;
    pad_top_ = pad_height_ / kSymCoef;
    pad_left_ = pad_width_ / kSymCoef;
    paddingA[kIndex0] = pad_front_;
    paddingA[kIndex1] = pad_top_;
    paddingA[kIndex2] = pad_left_;
  } else if (pad_mode_ == PadMode::VALID) {
    pad_depth_ = 0;
    pad_height_ = 0;
    pad_width_ = 0;
  } else {
    if (pad_list_.size() != kPadListSize) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the length of 'pad_list' must be 6, but got "
                        << pad_list_.size();
    }
    for (size_t idx = 0; idx < k3dDimSize; idx++) {
      paddingA[idx] = pad_list_[idx * kPadScale];
    }
  }
  std::string err_msg = "For '" + kernel_name_ + "', cudnnSetPoolingNdDescriptor failed";
  CHECK_CUDNN_RET_WITH_EXCEPT_NOTRACE(
    cudnnSetPoolingNdDescriptor(pooling_descriptor_, pooling_mode_, CUDNN_NOT_PROPAGATE_NAN, k3dDimSize, windowDimA,
                                paddingA, strideA),
    err_msg);
}

void PoolingGradGpuKernelMod::SetPoolingMode() {
  if (kernel_name_ == kAvgPool3DGrad) {
    pooling_mode_ =
      include_ ? CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING : CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    pad_value_ = 0.0;
  } else if (kernel_name_ == kAvgPoolGrad) {
    pooling_mode_ = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING;
    pad_value_ = 0.0;
  } else {
    pooling_mode_ = CUDNN_POOLING_MAX;
    pad_value_ = kSignedMinFloat;
  }
}

std::vector<int64_t> PoolingGradGpuKernelMod::GetEdgeKernelSize() {
  if (!ceil_mode_ && divisor_override_ == 0) {
    return {};
  }

  const size_t k3dSizeLowerLimit = 5;
  const size_t kIdxD = 2;
  const size_t kIdxH = 3;
  const size_t kIdxW = 4;
  const size_t kScale = 2;
  std::vector<int64_t> edge_kernel;
  if (window_me_.size() != k3dSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "kernel_size must be " << k3dSizeLowerLimit << "D, but got " << window_me_.size();
  }
  if (stride_me_.size() != k3dSizeLowerLimit) {
    MS_LOG(EXCEPTION) << "strides must be " << k3dSizeLowerLimit << "D, but got " << stride_me_.size();
  }

  kernel_size_ = {window_me_[kIdxD], window_me_[kIdxH], window_me_[kIdxW]};
  std::vector<int64_t> stride = {stride_me_[kIdxD], stride_me_[kIdxH], stride_me_[kIdxW]};
  std::vector<int64_t> shape_exclude_nc = {output_shape_[kIdxD], output_shape_[kIdxH], output_shape_[kIdxW]};

  const size_t kDim = shape_exclude_nc.size();
  if (pad_list_.size() != kDim * kScale) {
    MS_LOG(EXCEPTION) << "pad_list must be " << (kDim * kScale) << "D, but got " << pad_list_.size() << "D!";
  }

  for (size_t i = 0; i < kDim; ++i) {
    size_t l_index = kScale * i;
    size_t r_index = kScale * i + 1;

    int64_t len = shape_exclude_nc[i] + pad_list_[l_index] + pad_list_[r_index] - kernel_size_[i];
    int64_t padding_iv = FloatToLong(std::ceil(LongToFloat(len) / LongToFloat(stride[i]))) * stride[i] - len;
    int64_t padding_r = pad_list_[r_index] + padding_iv;
    if (padding_r > pad_list_[r_index] && padding_r < kernel_size_[i]) {
      edge_kernel.push_back(kernel_size_[i] - padding_iv);
    } else {
      edge_kernel.push_back(kernel_size_[i]);
    }
  }
  return edge_kernel;
}

void PoolingGradGpuKernelMod::SetFirstInputIndex(size_t input_num) {
  if (kernel_name_ == kAvgPool3DGrad && input_num == kAvgPool3DGradDynamicInputNum) {
    first_input_index_ = 1;
  }
}

std::map<std::string, std::vector<std::pair<KernelAttr, PoolingGradGpuKernelMod::PoolingGradFunc>>>
  PoolingGradGpuKernelMod::kernel_attr_map_ = {
    {kMaxPoolGrad,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &PoolingGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &PoolingGradGpuKernelMod::LaunchKernel<half>}}},
    {kMaxPool3DGrad,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &PoolingGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &PoolingGradGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &PoolingGradGpuKernelMod::LaunchKernel<double>}}},
    {kAvgPoolGrad,
     {{KernelAttr()
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddInputAttr(kNumberTypeFloat32)
         .AddOutputAttr(kNumberTypeFloat32),
       &PoolingGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddInputAttr(kNumberTypeFloat16)
         .AddOutputAttr(kNumberTypeFloat16),
       &PoolingGradGpuKernelMod::LaunchKernel<half>},
      {KernelAttr()
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddInputAttr(kNumberTypeFloat64)
         .AddOutputAttr(kNumberTypeFloat64),
       &PoolingGradGpuKernelMod::LaunchKernel<double>}}},
    {kAvgPool3DGrad,
     {{KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &PoolingGradGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &PoolingGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &PoolingGradGpuKernelMod::LaunchKernel<half>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
       &PoolingGradGpuKernelMod::LaunchKernel<double>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
       &PoolingGradGpuKernelMod::LaunchKernel<float>},
      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
       &PoolingGradGpuKernelMod::LaunchKernel<half>}}}};

std::vector<KernelAttr> PoolingGradGpuKernelMod::GetOpSupport() {
  auto iter = kernel_attr_map_.find(kernel_name_);
  if (iter == kernel_attr_map_.end()) {
    MS_LOG(ERROR)
      << "For 'PoolingGradGpuKernelMod', the kernel name must be in "
      << kernel::Map2Str<std::map, std::vector<std::pair<KernelAttr, PoolingGradGpuKernelMod::PoolingGradFunc>>>(
           kernel_attr_map_)
      << ", but got " << kernel_name_;
    return std::vector<KernelAttr>{};
  }
  std::vector<KernelAttr> support_list;
  (void)std::transform(iter->second.begin(), iter->second.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, PoolingGradFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPoolGrad,
                                 []() { return std::make_shared<PoolingGradGpuKernelMod>(kMaxPoolGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, MaxPool3DGrad,
                                 []() { return std::make_shared<PoolingGradGpuKernelMod>(kMaxPool3DGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AvgPoolGrad,
                                 []() { return std::make_shared<PoolingGradGpuKernelMod>(kAvgPoolGrad); });
MS_KERNEL_FACTORY_REG_BY_CREATOR(NativeGpuKernelMod, AvgPool3DGrad,
                                 []() { return std::make_shared<PoolingGradGpuKernelMod>(kAvgPool3DGrad); });
}  // namespace kernel
}  // namespace mindspore
