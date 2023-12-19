/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#include "utils/log_adapter.h"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
constexpr int kUpsampleNearest3DGpuGradInputsNum = 3;
constexpr int kUpsampleNearest3DGpuGradOutputsNum = 1;
}  // namespace
bool UpsampleNearest3DGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleNearest3DGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (auto ret = NativeGpuKernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> dy_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> dx_shape = outputs[kIndex0]->GetShapeVector();
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

  auto type = inputs[kIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(type);
  auto output_size_none = type->isa<TypeNone>();
  auto scales_opt = inputs[kIndex3]->GetOptionalValueWithCheck<std::vector<float>>();
  bool scales_none = !scales_opt.has_value();
  if (output_size_none == scales_none) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
    return KRET_RESIZE_FAILED;
  }

  if (!output_size_none) {
    scales_ = std::vector<float>(kIndex3, kValueZero);
  } else {
    scales_ = scales_opt.value();
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3DGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &workspace,
                                                     const std::vector<KernelTensor *> &outputs) {
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

#define UpsampleNearest3D_GRAD_GPU_KERNEL_REG(M_S, T)                    \
  std::make_pair(KernelAttr()                                            \
                   .AddInputAttr(M_S)                                    \
                   .AddInputAttr(kNumberTypeInt32)                       \
                   .AddOptionalInputAttr(kNumberTypeInt32)               \
                   .AddOptionalInputAttr(kNumberTypeFloat32)             \
                   .AddOutputAttr(M_S),                                  \
                 &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>),   \
    std::make_pair(KernelAttr()                                          \
                     .AddInputAttr(M_S)                                  \
                     .AddInputAttr(kNumberTypeInt32)                     \
                     .AddOptionalInputAttr(kNumberTypeInt64)             \
                     .AddOptionalInputAttr(kNumberTypeFloat32)           \
                     .AddOutputAttr(M_S),                                \
                   &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>), \
    std::make_pair(KernelAttr()                                          \
                     .AddInputAttr(M_S)                                  \
                     .AddInputAttr(kNumberTypeInt64)                     \
                     .AddOptionalInputAttr(kNumberTypeInt32)             \
                     .AddOptionalInputAttr(kNumberTypeFloat32)           \
                     .AddOutputAttr(M_S),                                \
                   &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>), \
    std::make_pair(KernelAttr()                                          \
                     .AddInputAttr(M_S)                                  \
                     .AddInputAttr(kNumberTypeInt64)                     \
                     .AddOptionalInputAttr(kNumberTypeInt64)             \
                     .AddOptionalInputAttr(kNumberTypeFloat32)           \
                     .AddOutputAttr(M_S),                                \
                   &UpsampleNearest3DGradGpuKernelMod::LaunchKernel<T>)

std::vector<std::pair<KernelAttr, UpsampleNearest3DGradGpuKernelMod::UpsampleNearest3DGradFunc>>
  UpsampleNearest3DGradGpuKernelMod::func_list_ = {
    // Get indent error when using a macro to generate 4 kernel_attr pair.
    // So add 4(each int means a int32 and int64) for each here...
    UpsampleNearest3D_GRAD_GPU_KERNEL_REG(kNumberTypeFloat16, half),
    UpsampleNearest3D_GRAD_GPU_KERNEL_REG(kNumberTypeFloat32, float),
    UpsampleNearest3D_GRAD_GPU_KERNEL_REG(kNumberTypeFloat64, double)};

std::vector<KernelAttr> UpsampleNearest3DGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleNearest3DGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleNearest3DGrad, UpsampleNearest3DGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
