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

#include "plugin/device/gpu/kernel/math/compare_and_bitpack_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include "include/curand.h"
#include "mindspore/core/ops/compareAndBitpack.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/compare_and_bitpack_impl.cuh"
#include "include/common/utils/anfalgo.h"

namespace mindspore {
namespace kernel {
constexpr size_t kBitpack = 8;
bool CompareAndBitpackGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                         const std::vector<KernelTensorPtr> &inputs,
                                         const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it got empty inputs or outputs, which is invalid.";
    return false;
  }
  kernel_ptr_ = std::make_shared<ops::CompareAndBitpack>(base_operator->GetPrim());
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the kernel type should be in "
                  << "[int8, int16, int32, int64, float16, float32, float64, bool], but got: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  x_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  threshold_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).first);
  cudnn_handle_ = device::gpu::GPUDeviceManager::GetInstance().GetCudnnHandle();
  return true;
}

int CompareAndBitpackGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                          const std::vector<KernelTensorPtr> &inputs,
                                          const std::vector<KernelTensorPtr> &outputs,
                                          const std::map<uint32_t, tensor::TensorPtr> &) {
  ResetResource();
  for (const auto &input : inputs) {
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto x_long_shape = inputs.at(kIndex0)->GetShapeVector();
  std::vector<size_t> x_shape;
  (void)std::transform(x_long_shape.begin(), x_long_shape.end(), std::back_inserter(x_shape), LongToSize);
  for (size_t i = 0; i < x_shape.size(); i++) {
    x_count_ *= x_shape[i];
  }
  y_count_ = x_count_ / kBitpack;
  size_t x_size = x_count_ * x_unit_size_;
  input_size_list_.emplace_back(x_size);
  size_t threshold_size = threshold_unit_size_;
  input_size_list_.emplace_back(threshold_size);
  size_t output_size = y_count_ * sizeof(uint8_t);
  output_size_list_.emplace_back(output_size);
  size_t workspace_size = 0;
  workspace_size_list_.emplace_back(workspace_size);
  return KRET_OK;
}

void CompareAndBitpackGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  x_count_ = 1;
  input_size_list_.clear();
  output_size_list_.clear();
  workspace_size_list_.clear();
}

template <typename T>
bool CompareAndBitpackGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                 const std::vector<AddressPtr> &workspace,
                                                 const std::vector<AddressPtr> &outputs) {
  T *x = GetDeviceAddress<T>(inputs, kIndex0);
  T *threshold = GetDeviceAddress<T>(inputs, kIndex1);
  uint8_t *y = GetDeviceAddress<uint8_t>(outputs, kIndex0);
  CalCompareAndBitpack(x, threshold, y, y_count_, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
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
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, CompareAndBitpack, CompareAndBitpackGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
