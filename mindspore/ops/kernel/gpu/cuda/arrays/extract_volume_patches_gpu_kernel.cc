/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/gpu/cuda/arrays/extract_volume_patches_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "kernel/gpu/cuda_impl/cuda_ops/extract_volume_patches_impl.cuh"

namespace mindspore {
namespace kernel {
constexpr size_t kDimSize5 = 5;
constexpr size_t kFormatNCDHWIndexC = 1;
constexpr size_t kFormatNCDHWIndexD = 2;
constexpr size_t kFormatNCDHWIndexH = 3;
constexpr size_t kFormatNCDHWIndexW = 4;

bool ExtractVolumePatchesGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  kernel_size_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
  strides_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr("strides"));
  padding_ = GetValue<std::string>(primitive_->GetAttr("padding"));
  size_t kernel_size_dims = kernel_size_.size();
  size_t strides_dims = strides_.size();
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return KRET_RESIZE_FAILED;
  }
  if (kernel_size_dims != kDimSize5) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'kernel_size' must be equal to 5, but got "
                  << kernel_size_dims << ".";
    return false;
  }
  if (strides_dims != kDimSize5) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'strides' must be equal to 5, but got "
                  << strides_dims << ".";
    return false;
  }
  if (padding_ != "VALID" && padding_ != "valid" && padding_ != "SAME" && padding_ != "same") {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', padding_ must be VALID, valid, SAME or same, but got " << padding_
                  << ".";
    return false;
  }
  return true;
}

int ExtractVolumePatchesGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  size_t input_shape_dims = input_shape_.size();
  size_t output_shape_dims = output_shape_.size();
  // check parameter
  if (input_shape_dims != kDimSize5) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'input_shape' must be equal to 5, but got "
                  << input_shape_dims << ".";
    return KRET_RESIZE_FAILED;
  }
  if (output_shape_dims != kDimSize5) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'output_shape' must be equal to 5, but got "
                  << output_shape_dims << ".";
    return KRET_RESIZE_FAILED;
  }
  ksize_d_ = kernel_size_[kFormatNCDHWIndexD];
  ksize_h_ = kernel_size_[kFormatNCDHWIndexH];
  ksize_w_ = kernel_size_[kFormatNCDHWIndexW];
  stride_d_ = strides_[kFormatNCDHWIndexD];
  stride_h_ = strides_[kFormatNCDHWIndexH];
  stride_w_ = strides_[kFormatNCDHWIndexW];
  input_channel_ = input_shape_[kFormatNCDHWIndexC];
  input_depth_ = input_shape_[kFormatNCDHWIndexD];
  input_height_ = input_shape_[kFormatNCDHWIndexH];
  input_width_ = input_shape_[kFormatNCDHWIndexW];
  output_depth_ = output_shape_[kFormatNCDHWIndexD];
  output_height_ = output_shape_[kFormatNCDHWIndexH];
  output_width_ = output_shape_[kFormatNCDHWIndexW];

  if (padding_.compare("VALID") == 0 || padding_.compare("valid") == 0) {
    pad_head_ = 0;
    pad_top_ = 0;
    pad_left_ = 0;
  }
  if (padding_.compare("SAME") == 0 || padding_.compare("same") == 0) {
    constexpr int64_t zero_value = 0;
    constexpr int64_t kMidDividend = 2;
    pad_head_ = std::max(zero_value, ((output_depth_ - 1) * stride_d_ + ksize_d_ - input_depth_) / kMidDividend);
    pad_top_ = std::max(zero_value, ((output_height_ - 1) * stride_h_ + ksize_h_ - input_height_) / kMidDividend);
    pad_left_ = std::max(zero_value, ((output_width_ - 1) * stride_w_ + ksize_w_ - input_width_) / kMidDividend);
  }
  output_size_ = 1;
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_size_ *= output_shape_[i];
  }
  d_stride_ = ksize_h_ * ksize_w_;
  h_stride_ = ksize_h_;
  w_stride_ = ksize_w_;
  patch_stride_ = output_depth_ * output_height_ * output_width_;
  other_stride_ = patch_stride_ * ksize_d_ * ksize_h_ * ksize_w_ * input_channel_;
  chan_input_stride_ = input_depth_ * input_height_ * input_width_;
  dep_input_stride_ = input_height_ * input_width_;
  row_input_stride_ = input_width_;
  patch_input_stride_ = input_channel_ * input_depth_ * input_height_ * input_width_;
  MS_EXCEPTION_IF_ZERO("other stride", other_stride_);
  need_batch_ = (output_size_ - 1) / other_stride_;
  return KRET_OK;
}

template <typename T>
bool ExtractVolumePatchesGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &workspace,
                                                    const std::vector<KernelTensor *> &outputs) {
  T *input_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  auto status = CalExtractVolumePatches(
    output_size_, stride_d_, stride_h_, stride_w_, output_depth_, output_height_, output_width_, need_batch_, d_stride_,
    h_stride_, w_stride_, patch_stride_, other_stride_, input_channel_, input_depth_, input_height_, input_width_,
    pad_head_, pad_top_, pad_left_, chan_input_stride_, dep_input_stride_, row_input_stride_, patch_input_stride_,
    input_ptr, output_ptr, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

using FuncList = std::vector<std::pair<KernelAttr, ExtractVolumePatchesGpuKernelMod::KernelRunFunc>>;
const FuncList &ExtractVolumePatchesGpuKernelMod::GetFuncList() const {
  static const FuncList func_list_ = {{KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<double>},
                                      {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<float>},
                                      {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<half>},
                                      {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<int64_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<int32_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeInt16),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<int16_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeInt8),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<int8_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeUInt64),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<uint64_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeUInt32),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<uint32_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeUInt16),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<uint16_t>},
                                      {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeUInt8),
                                       &ExtractVolumePatchesGpuKernelMod::LaunchKernel<uint8_t>}};
  return func_list_;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, ExtractVolumePatches, ExtractVolumePatchesGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
