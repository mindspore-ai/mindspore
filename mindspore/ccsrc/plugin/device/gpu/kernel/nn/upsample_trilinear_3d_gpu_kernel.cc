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

#include "plugin/device/gpu/kernel/nn/upsample_trilinear_3d_gpu_kernel.h"
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
#include "mindspore/core/ops/upsample_trilinear_3d.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_trilinear_3d_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
const double kValueZero = 0.;
constexpr int kInputsNum = 2;
constexpr int kOutputsNum = 1;
}  // namespace
bool UpsampleTrilinear3DGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  align_corners_ = GetValue<bool>(primitive_->GetAttr(ops::kAlignCorners));
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int UpsampleTrilinear3DGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  std::vector<int64_t> input_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_shape = outputs[kIndex0]->GetShapeVector();
  n_ = input_shape[kIndex0];
  c_ = input_shape[kIndex1];
  input_d_ = input_shape[kIndex2];
  input_h_ = input_shape[kIndex3];
  input_w_ = input_shape[kIndex4];
  output_d_ = output_shape[kIndex2];
  output_h_ = output_shape[kIndex3];
  output_w_ = output_shape[kIndex4];

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

template <typename T, typename S>
bool UpsampleTrilinear3DGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &workspace,
                                                   const std::vector<KernelTensor *> &outputs) {
  auto x_ptr = GetDeviceAddress<T>(inputs, kIndex0);
  MS_EXCEPTION_IF_NULL(x_ptr);
  auto y_ptr = GetDeviceAddress<T>(outputs, kIndex0);
  MS_EXCEPTION_IF_NULL(y_ptr);

  const S depth_scale = AreaPixelComputeScale<S>(input_d_, output_d_, align_corners_, scales_[kIndex0]);
  const S height_scale = AreaPixelComputeScale<S>(input_h_, output_h_, align_corners_, scales_[kIndex1]);
  const S width_scale = AreaPixelComputeScale<S>(input_w_, output_w_, align_corners_, scales_[kIndex2]);

  auto status = CalUpsampleTrilinear3D<T, S>(x_ptr, n_, c_, input_d_, input_h_, input_w_, output_d_, output_h_,
                                             output_w_, depth_scale, height_scale, width_scale, align_corners_, y_ptr,
                                             device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

#define UpsampleTrilinear3D_GPU_KERNEL_REG(M_S, M_T, S, T)             \
  KernelAttr().AddInputAttr(M_S).AddInputAttr(M_T).AddOutputAttr(M_S), \
    &UpsampleTrilinear3DGpuKernelMod::LaunchKernel<S, T>

std::vector<std::pair<KernelAttr, UpsampleTrilinear3DGpuKernelMod::UpsampleTrilinear3DFunc>>
  UpsampleTrilinear3DGpuKernelMod::func_list_ = {
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt32, half, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt32, float, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt32, double, double)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeInt64, half, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeInt64, float, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeInt64, double, double)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat16, kNumberTypeFloat32, half, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat32, kNumberTypeFloat32, float, float)},
    {UpsampleTrilinear3D_GPU_KERNEL_REG(kNumberTypeFloat64, kNumberTypeFloat32, double, double)},
};

std::vector<KernelAttr> UpsampleTrilinear3DGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, UpsampleTrilinear3DFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, UpsampleTrilinear3D, UpsampleTrilinear3DGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
