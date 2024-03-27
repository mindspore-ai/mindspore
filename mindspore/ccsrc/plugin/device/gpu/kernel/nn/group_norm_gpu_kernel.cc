/**
 * Copyright 2020-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/gpu/kernel/nn/group_norm_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/group_norm_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGroupNormInputsNum = 5;
constexpr size_t kGroupNormOutputsNum = 3;
constexpr size_t kNumberTwo = 2;
}  // namespace
bool GroupNormGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GroupNormGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid GroupNormGpuKernelMod input size!";
  }

  const auto &x_shape = inputs[kIndex0]->GetShapeVector();
  if (x_shape.size() < 2) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the dims of input tesnor must be not less than 2 "
                      << "but got: " << x_shape.size();
  }

  auto batch = x_shape[0];
  auto num_groups = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  HxW_ = LongToSize((x_shape.size() == kNumberTwo)
                      ? 1
                      : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>()));
  eps_ = inputs[kIndex4]->GetValueWithCheck<float_t>();
  num_channel_ = LongToSize(x_shape[1]);
  inner_size_ = LongToSize(num_channel_ * HxW_ / num_groups);
  outter_size_ = LongToSize(batch * num_groups);

  if (num_channel_ % num_groups != 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'num_channels' must be divided by 'num_groups', "
                      << "but got 'num_channels': " << num_channel_ << " ,'num_groups': " << num_groups;
  }

  return ret;
}

bool GroupNormGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGroupNormInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGroupNormOutputsNum, kernel_name_);
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  kernel_func_(this, inputs, outputs);
  return true;
}

template <typename T>
void GroupNormGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  size_t f_size = sizeof(T);
  if (inputs[kIndex2]->size() != f_size * LongToUlong(num_channel_) ||
      inputs[kIndex3]->size() != f_size * LongToUlong(num_channel_)) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the product of gamma and beta's shape must be " << num_channel_;
  }
  auto x = GetDeviceAddress<T>(inputs, kIndex0);
  auto gamma = GetDeviceAddress<T>(inputs, kIndex2);
  auto beta = GetDeviceAddress<T>(inputs, kIndex3);
  auto y = GetDeviceAddress<T>(outputs, kIndex0);
  auto mean = GetDeviceAddress<T>(outputs, kIndex1);
  auto rstd = GetDeviceAddress<T>(outputs, kIndex2);
  MS_EXCEPTION_IF_NULL(x);
  MS_EXCEPTION_IF_NULL(gamma);
  MS_EXCEPTION_IF_NULL(beta);
  MS_EXCEPTION_IF_NULL(y);
  MS_EXCEPTION_IF_NULL(mean);
  MS_EXCEPTION_IF_NULL(rstd);

  auto status =
    GroupNorm(outter_size_, inner_size_, num_channel_, HxW_, eps_, x, gamma, beta, y, mean, rstd, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
}

std::vector<std::pair<KernelAttr, GroupNormGpuKernelMod::KernelFunc>> GroupNormGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GroupNormGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GroupNormGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GroupNormGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> GroupNormGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GroupNorm, GroupNormGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
