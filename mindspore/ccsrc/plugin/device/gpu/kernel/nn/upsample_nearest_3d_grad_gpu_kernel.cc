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

#include "plugin/device/gpu/kernel/nn/upsample_nearest_3d_grad_gpu_kernel.h"
#include <algorithm>
#include <functional>
#include <iostream>
#include <string>
#include <utility>
#include "kernel/kernel_get_value.h"
#include "kernel/ops_utils.h"
#include "mindspore/core/abstract/utils.h"
#include "mindspore/core/ops/grad/upsample_nearest_3d_grad.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_grad_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
constexpr int kUpsampleNearest3DGpuGradInputsNum = 3;
constexpr int kUpsampleNearest3DGpuGradOutputsNum = 1;
}  // namespace
bool UpsampleNearest3DGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                             const std::vector<KernelTensorPtr> &inputs,
                                             const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleNearest3DGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                              const std::vector<KernelTensorPtr> &inputs,
                                              const std::vector<KernelTensorPtr> &outputs,
                                              const std::map<uint32_t, tensor::TensorPtr> &) {
  if (auto ret = NativeGpuKernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  std::vector<int64_t> dy_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<int64_t> dx_shape = outputs.at(kIndex0)->GetShapeVector();
  n_ = dy_shape[kIndex0];
  c_ = dy_shape[kIndex1];
  // input
  dy_d_ = dy_shape[kIndex2];
  dy_h_ = dy_shape[kIndex3];
  dy_w_ = dy_shape[kIndex4];
  // output
  dx_d_ = dx_shape[kIndex2];
  dx_h_ = dx_shape[kIndex3];
  dx_w_ = dx_shape[kIndex4];
  // none list
  MS_EXCEPTION_IF_NULL(base_operator);
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex3)) {
    scales_ = std::vector<double>(kIndex3, kValueZero);
  } else {
    if (!TryGetFloatValue(inputs, kIndex2, kernel_name_, &scales_)) {
      MS_LOG(EXCEPTION) << "For " << kernel_name_ << " can't get scales input! ";
    }
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3DGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                     const std::vector<AddressPtr> &workspace,
                                                     const std::vector<AddressPtr> &outputs) {
  auto dy = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(dy);
  auto dx = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(dx);

  const float d_scale = ComputeScalesBackward<float>(scales_[kIndex0], dy_d_, dx_d_);
  const float h_scale = ComputeScalesBackward<float>(scales_[kIndex1], dy_h_, dx_h_);
  const float w_scale = ComputeScalesBackward<float>(scales_[kIndex2], dy_w_, dx_w_);

  auto status = CalUpsampleNearest3DGrad(dy, n_, c_, dy_d_, dy_h_, dy_w_, dx_d_, dx_h_, dx_w_, d_scale, h_scale,
                                         w_scale, dx, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(M_S, M_T, T)                                      \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt32).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>
#define UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(M_S, M_T, T)                                      \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(kNumberTypeInt64).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>

std::vector<std::pair<KernelAttr, UpsampleNearest3DGradGpuKernelMod::UpsampleNearest3DGradFunc>>
  UpsampleNearest3DGradGpuKernelMod::func_list_ = {
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt32, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeInt64, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat16, kNumberTypeFloat32, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT32_REG(kNumberTypeFloat64, kNumberTypeFloat32, double)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt32, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt32, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt32, double)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeInt64, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeInt64, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeInt64, double)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat16, kNumberTypeFloat32, half)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat32, kNumberTypeFloat32, float)},
    {UpsampleNearest3D_GRAD_GPU_KERNEL_INT64_REG(kNumberTypeFloat64, kNumberTypeFloat32, double)}};

std::vector<KernelAttr> UpsampleNearest3DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleNearest3DGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleNearest3DGrad, UpsampleNearest3DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
