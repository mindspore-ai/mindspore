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
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <utility>
#include <functional>
#include "abstract/utils.h"
#include "kernel/common_utils.h"
#include "mindspore/core/ops/upsample_nearest_3d.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/upsample_nearest_3d_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
const float kValueZero = 0.;
}  // namespace
bool UpsampleNearest3dGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  MS_EXCEPTION_IF_NULL(base_operator);
  kernel_name_ = base_operator->name();
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

int UpsampleNearest3dGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  if (int ret = KernelMod::Resize(base_operator, inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  input_shape_.clear();
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(input_shape.begin(), input_shape.end(), std::back_inserter(input_shape_), LongToInt);
  output_shape_.clear();
  auto output_shape = outputs.at(kIndex0)->GetShapeVector();
  (void)std::transform(output_shape.begin(), output_shape.end(), std::back_inserter(output_shape_), LongToInt);
  none_list_ = GetValue<std::vector<int64_t>>(base_operator->GetAttr(kAttrNoneList));
  if (none_list_.size() != kIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', only one of output_size or scales should be specified.";
  }
  return KRET_OK;
}

template <typename T>
bool UpsampleNearest3dGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), 2, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), 1, kernel_name_);
  if (none_list_[kIndex0] == static_cast<int64_t>(kIndex2)) {
    scale_factors_ = std::vector<float>(kIndex3, kValueZero);
  } else {
    auto scale_factors_device = GetDeviceAddress<float>(inputs, kIndex1);
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
      cudaMemcpyAsync(reinterpret_cast<void *>(scale_factors_.data()), reinterpret_cast<void *>(scale_factors_device),
                      input_size_list_[kIndex1], cudaMemcpyDeviceToHost, reinterpret_cast<cudaStream_t>(cuda_stream_)),
      "For '" << kernel_name_ << "', "
              << "cudaMemcpy input 'scales' to host failed.");
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaDeviceSynchronize(), "cudaDeviceSyncFailed - " + kernel_name_);
  }
  const float scale_d =
    ComputeScales<float>(static_cast<double>(scale_factors_[kIndex0]), input_shape_[kIndex2], output_shape_[kIndex2]);
  const float scale_h =
    ComputeScales<float>(static_cast<double>(scale_factors_[kIndex1]), input_shape_[kIndex3], output_shape_[kIndex3]);
  const float scale_w =
    ComputeScales<float>(static_cast<double>(scale_factors_[kIndex2]), input_shape_[kIndex4], output_shape_[kIndex4]);
  auto input = reinterpret_cast<T *>(inputs.at(kIndex0)->addr);
  auto output = reinterpret_cast<T *>(outputs.at(kIndex0)->addr);
  auto status = CalUpsampleNearest3d<T>(input, input_shape_[kIndex0], input_shape_[kIndex1], input_shape_[kIndex2],
                                        input_shape_[kIndex3], input_shape_[kIndex4], output_shape_[kIndex2],
                                        output_shape_[kIndex3], output_shape_[kIndex4], scale_d, scale_h, scale_w,
                                        output, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_LAUNCH_STATUS(status, kernel_name_);
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
