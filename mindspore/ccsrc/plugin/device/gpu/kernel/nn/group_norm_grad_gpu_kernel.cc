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

#include "plugin/device/gpu/kernel/nn/group_norm_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <numeric>
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/group_norm_grad_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kGroupNormGradInputsNum = 9;
constexpr size_t kGroupNormGradOutputsNum = 3;
constexpr size_t kNumberTwo = 2;
}  // namespace

bool GroupNormGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For GPU '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int GroupNormGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  if (ret != 0) {
    return ret;
  }
  if (inputs.empty()) {
    MS_LOG(EXCEPTION) << "Invalid GroupNormGradGpuKernelMod input size!";
  }

  auto x_shape = inputs[kIndex1]->GetShapeVector();

  batch_ = LongToSize(x_shape[kIndex0]);
  num_channel_ = LongToSize(x_shape[kIndex1]);
  HxW_ = LongToSize((x_shape.size() == kNumberTwo)
                      ? 1
                      : std::accumulate(x_shape.begin() + kIndex2, x_shape.end(), 1, std::multiplies<int64_t>()));
  num_groups_ = LongToSize(inputs[kIndex5]->GetValueWithCheck<int64_t>());

  const size_t dscale_shape_size = LongToSize(batch_ * num_channel_) * sizeof(float);
  const size_t dbias_shape_size = LongToSize(batch_ * num_channel_) * sizeof(float);

  workspace_size_list_.clear();
  workspace_size_list_ = {dscale_shape_size, dbias_shape_size};

  return ret;
}

bool GroupNormGradGpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kGroupNormGradInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kGroupNormGradOutputsNum, kernel_name_);
  cuda_stream_ = reinterpret_cast<cudaStream_t>(stream_ptr);
  kernel_func_(this, inputs, workspace, outputs);
  return true;
}

template <typename T>
void GroupNormGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  auto dy = GetDeviceAddress<T>(inputs, kIndex0);
  auto x = GetDeviceAddress<T>(inputs, kIndex1);
  auto mean = GetDeviceAddress<T>(inputs, kIndex2);
  auto rstd = GetDeviceAddress<T>(inputs, kIndex3);
  auto gamma = GetDeviceAddress<T>(inputs, kIndex4);
  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  auto dg = GetDeviceAddress<T>(outputs, kIndex1);
  auto db = GetDeviceAddress<T>(outputs, kIndex2);
  auto dscale = GetDeviceAddress<float>(workspace, kIndex0);
  auto dbias = GetDeviceAddress<float>(workspace, kIndex1);

  auto status = GroupNormGrad(batch_, num_channel_, HxW_, num_groups_, dy, x, mean, rstd, gamma, dx, dg, db, dscale,
                              dbias, cuda_stream_);
  CHECK_CUDA_STATUS(status, kernel_name_);
}

std::vector<std::pair<KernelAttr, GroupNormGradGpuKernelMod::KernelFunc>> GroupNormGradGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kNumberTypeFloat16)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16)
     .AddOutputAttr(kNumberTypeFloat16),
   &GroupNormGradGpuKernelMod::LaunchKernel<half>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &GroupNormGradGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &GroupNormGradGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> GroupNormGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, GroupNormGrad, GroupNormGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
