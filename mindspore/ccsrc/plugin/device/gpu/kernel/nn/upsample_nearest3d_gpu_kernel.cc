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

#include "plugin/device/gpu/kernel/nn/upsample_nearest3d_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "abstract/utils.h"
#include "kernel/kernel_get_value.h"
#include "kernel/ops_utils.h"
#include "ops/upsample_nearest_3d.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
}  // namespace
bool UpsampleNearest3dGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleNearest3dGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_ = inputs[kIndex0]->GetShapeVector();
  output_shape_ = outputs[kIndex0]->GetShapeVector();

  none_list_ = GetValue<std::vector<int64_t>>(primitive_->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex2)) {
    scales_ = std::vector<float>(kIndex3, kValueZero);
  } else {
    scales_ = inputs[kIndex1]->GetValueWithCheck<std::vector<float>>();
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3dGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(input);
  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(output);

  const float scale_d = ComputeScales<float>(scales_[kIndex0], input_shape_[kIndex2], output_shape_[kIndex2]);
  const float scale_h = ComputeScales<float>(scales_[kIndex1], input_shape_[kIndex3], output_shape_[kIndex3]);
  const float scale_w = ComputeScales<float>(scales_[kIndex2], input_shape_[kIndex4], output_shape_[kIndex4]);
  auto status = CalUpsampleNearest3d<T>(input, input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2],
                                        input_shape_[kIndex3], input_shape_[kIndex4], output_shape_[kIndex2],
                                        output_shape_[kIndex3], output_shape_[kIndex4], scale_d, scale_h, scale_w,
                                        output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define UpsampleNearest3D_GPU_KERNEL_REG(M_S, M_T, S) \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(M_T).AddOutputAttr(M_S), &UpsampleNearest3dGpuKernelMod::LaunchKernel<S>

std::vector<std::pair<KernelAttr, UpsampleNearest3dGpuKernelMod::UpsampleNearest3dFunc>>
  UpsampleNearest3dGpuKernelMod::func_list_ = {
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeInt32, uint8_t)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt32, half)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeInt64, uint8_t)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt64, half)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeUInt8, kNumberTypeFloat32, uint8_t)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeFloat32, half)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
    {UpsampleNearest3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeFloat32, double)},
};

std::vector<KernelAttr> UpsampleNearest3dGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleNearest3dFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleNearest3D, UpsampleNearest3dGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
