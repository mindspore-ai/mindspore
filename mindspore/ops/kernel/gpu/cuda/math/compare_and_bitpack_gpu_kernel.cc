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

#include "kernel/gpu/cuda/math/compare_and_bitpack_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/ops/infer/compare_and_bitpack.h"
#include "kernel/gpu/cuda_impl/cuda_ops/compare_and_bitpack_impl.cuh"
#include "include/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr size_t kBitpack = 8;
bool CompareAndBitpackGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in "
                  << "[int8, int16, int32, int64, float16, float32, float64, bool], but got: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  x_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  threshold_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  return true;
}

int CompareAndBitpackGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  ResetResource();
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  auto x_long_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> x_shape;
  (void)std::transform(x_long_shape.begin(), x_long_shape.end(), std::back_inserter(x_shape), LongToSize);
  for (size_t i = 0; i < x_shape.size(); i++) {
    x_count_ *= x_shape[i];
  }
  y_count_ = x_count_ / kBitpack;
  size_t output_size = y_count_ * sizeof(uint8_t);
  output_size_list_.emplace_back(output_size);
  size_t workspace_size = 0;
  workspace_size_list_.emplace_back(workspace_size);
  return KRET_OK;
}

void CompareAndBitpackGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  x_count_ = 1;
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool CompareAndBitpackGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &workspace,
                                                 const std::vector<KernelTensor *> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  T *threshold = GetDeviceAddress<T>(inputs, kIndex1);
  uint8_t *y = GetDeviceAddress<uint8_t>(outputs, kIndex0);
  auto status =
    CalCompareAndBitpack(x, threshold, y, y_count_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, CompareAndBitpackGpuKernelMod::CompareAndBitpackFunc>>
  CompareAndBitpackGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt8).AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt16).AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<int32_t>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeBool).AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeUInt8),
     &CompareAndBitpackGpuKernelMod::LaunchKernel<bool>}};

std::vector<KernelAttr> CompareAndBitpackGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, CompareAndBitpackFunc> &pair) { return pair.first; });
  return support_list;
}
}  // namespace kernel
}  // namespace mindspore
