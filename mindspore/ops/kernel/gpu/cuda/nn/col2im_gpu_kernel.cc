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

#include "kernel/gpu/cuda/nn/col2im_gpu_kernel.h"
#include <unordered_map>
#include <string>
#include <algorithm>
#include <limits>
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/col2im_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr int kCol2ImInputsNum = 2;
constexpr int kPaddingDirection = 2;
}  // namespace

void Col2ImFwdGpuKernelMod::ResetResource() noexcept {
  batch_size_ = 0;
  channels_ = 0;
  out_height_ = 0;
  out_width_ = 0;
  in_height_ = 0;
  in_width_ = 0;
  pad_height_ = 0;
  pad_width_ = 0;
  kernel_height_ = 0;
  kernel_width_ = 0;
  stride_height_ = 0;
  stride_width_ = 0;
  dilation_height_ = 0;
  dilation_width_ = 0;
  is_null_input_ = false;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T, typename S>
bool Col2ImFwdGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs) {
  T *input_addr = GetDeviceAddress<T>(inputs, kIndex0);
  T *output_addr = GetDeviceAddress<T>(outputs, kIndex0);
  Col2Im<T, S>(input_addr, batch_size_, channels_, out_height_, out_width_, in_height_, in_width_, kernel_height_,
               kernel_width_, pad_height_, pad_width_, stride_height_, stride_width_, dilation_height_, dilation_width_,
               output_addr, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

bool Col2ImFwdGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int Col2ImFwdGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.size() != kCol2ImInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be equal 2.";
    return KRET_RESIZE_FAILED;
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  batch_size_ = static_cast<uint32_t>(input_shape[kIndex0]);
  channels_ = static_cast<uint32_t>(input_shape[kIndex1]);
  out_height_ = static_cast<uint32_t>(output_shape[kIndex2]);
  out_width_ = static_cast<uint32_t>(output_shape[kIndex3]);
  auto kernel_size = GetValue<std::vector<int64_t>>(primitive_->GetAttr("kernel_size"));
  auto dilation = GetValue<std::vector<int64_t>>(primitive_->GetAttr("dilation"));
  auto padding = GetValue<std::vector<int64_t>>(primitive_->GetAttr("padding"));
  auto stride = GetValue<std::vector<int64_t>>(primitive_->GetAttr("stride"));
  std::unordered_map<std::string, std::vector<int64_t>> to_check{
    {"kernel_size", kernel_size}, {"dilation", dilation}, {"padding", padding}, {"stride", stride}};
  for (const auto &[name, vec] : to_check) {
    if (std::any_of(vec.begin(), vec.end(), [](int64_t x) { return x > std::numeric_limits<uint32_t>::max(); })) {
      MS_LOG(ERROR) << "For '" << kernel_name_ << "', " << name << " value overflow.";
      return KRET_RESIZE_FAILED;
    }
  }
  pad_height_ = static_cast<uint32_t>(padding[kIndex0]);
  pad_width_ = static_cast<uint32_t>(padding[kIndex1]);
  kernel_height_ = static_cast<uint32_t>(kernel_size[kIndex0]);
  kernel_width_ = static_cast<uint32_t>(kernel_size[kIndex1]);
  stride_height_ = static_cast<uint32_t>(stride[kIndex0]);
  stride_width_ = static_cast<uint32_t>(stride[kIndex1]);
  dilation_height_ = static_cast<uint32_t>(dilation[kIndex0]);
  dilation_width_ = static_cast<uint32_t>(dilation[kIndex1]);
  in_height_ = static_cast<uint32_t>(
    (out_height_ + kPaddingDirection * pad_height_ - (dilation_height_ * (kernel_height_ - 1) + 1)) / stride_height_ +
    1);
  in_width_ = static_cast<uint32_t>(
    (out_width_ + kPaddingDirection * pad_width_ - (dilation_width_ * (kernel_width_ - 1) + 1)) / stride_width_ + 1);
  return KRET_OK;
}

std::vector<std::pair<KernelAttr, Col2ImFwdGpuKernelMod::Col2ImFunc>> Col2ImFwdGpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
   &Col2ImFwdGpuKernelMod::LaunchKernel<float, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
   &Col2ImFwdGpuKernelMod::LaunchKernel<half, float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
   &Col2ImFwdGpuKernelMod::LaunchKernel<double, double>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex64).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex64),
   &Col2ImFwdGpuKernelMod::LaunchKernel<utils::Complex<float>, utils::Complex<float>>},
  {KernelAttr().AddInputAttr(kNumberTypeComplex128).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeComplex128),
   &Col2ImFwdGpuKernelMod::LaunchKernel<utils::Complex<double>, utils::Complex<double>>},
};

std::vector<KernelAttr> Col2ImFwdGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, Col2ImFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Col2Im, Col2ImFwdGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
