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

#include "kernel/gpu/cuda/sparse/sparse_reorder_gpu_kernel.h"
#include <functional>
#include <utility>
#include <string>
#include <algorithm>
#include <memory>
#include <complex>
#include "include/curand.h"
#include "mindspore/ops/infer/sparse_reorder.h"
#include "kernel/gpu/cuda_impl/cuda_ops/sparse_reorder_impl.cuh"

namespace mindspore {
namespace kernel {
bool SparseReorderGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).dtype);
  values_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex1).dtype);
  shape_unit_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex2).dtype);
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  return true;
}

int SparseReorderGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  input_elements_ = 0;
  output_size_list_.clear();
  workspace_size_list_.clear();
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  auto input_shape = inputs.at(kIndex0)->GetShapeVector();
  input_elements_ = std::accumulate(input_shape.begin(), input_shape.end(), size_t(1), std::multiplies<size_t>());
  num_elems_ = static_cast<int>(input_shape.at(0));
  num_dims_ = static_cast<int>(input_shape.at(1));
  auto values_shape = inputs.at(kIndex1)->GetShapeVector();
  values_elements_ = std::accumulate(values_shape.begin(), values_shape.end(), size_t(1), std::multiplies<size_t>());
  auto shape_shape = inputs.at(kIndex2)->GetShapeVector();
  shape_elements_ = std::accumulate(shape_shape.begin(), shape_shape.end(), size_t(1), std::multiplies<size_t>());
  auto output_indices_shape = outputs.at(kIndex0)->GetShapeVector();
  output_indices_elements_ = SizeOf(output_indices_shape);
  auto output_values_shape = outputs.at(kIndex1)->GetShapeVector();
  output_values_elements_ = SizeOf(output_values_shape);
  if (input_elements_ == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' input size must be greater than zero.";
    return KRET_RESIZE_FAILED;
  }

  size_t output_indices_size = output_indices_elements_ * unit_size_;
  size_t output_values_size = output_values_elements_ * values_unit_size_;
  size_t workspace_size = num_elems_ * unit_size_;
  output_size_list_.push_back(output_indices_size);
  output_size_list_.push_back(output_values_size);
  workspace_size_list_.push_back(workspace_size);
  workspace_size_list_.push_back(workspace_size);
  workspace_size_list_.push_back(sizeof(int));
  return KRET_OK;
}

template <typename T>
bool SparseReorderGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs) {
  int64_t *indices = GetDeviceAddress<int64_t>(inputs, kIndex0);
  T *values = GetDeviceAddress<T>(inputs, kIndex1);
  int64_t *shape = GetDeviceAddress<int64_t>(inputs, kIndex2);
  int64_t *y_indices = GetDeviceAddress<int64_t>(outputs, kIndex0);
  T *y_values = GetDeviceAddress<T>(outputs, kIndex1);
  int64_t *flat_indices = GetDeviceAddress<int64_t>(workspace, kIndex0);
  int64_t *permutation_data = GetDeviceAddress<int64_t>(workspace, kIndex1);
  int *check_flag = GetDeviceAddress<int32_t>(workspace, kIndex2);
  auto status = SparseReorder(num_elems_, num_dims_, indices, values, shape, y_indices, y_values, flat_indices,
                              permutation_data, check_flag, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

std::vector<std::pair<KernelAttr, SparseReorderGpuKernelMod::SparseReorderFunc>> SparseReorderGpuKernelMod::func_list_ =
  {{KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeBool)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeBool),
    &SparseReorderGpuKernelMod::LaunchKernel<bool>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt8),
    &SparseReorderGpuKernelMod::LaunchKernel<int8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt16),
    &SparseReorderGpuKernelMod::LaunchKernel<int16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt32),
    &SparseReorderGpuKernelMod::LaunchKernel<int32_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64),
    &SparseReorderGpuKernelMod::LaunchKernel<int64_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt8)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt8),
    &SparseReorderGpuKernelMod::LaunchKernel<uint8_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeUInt16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeUInt16),
    &SparseReorderGpuKernelMod::LaunchKernel<uint16_t>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat16)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat16),
    &SparseReorderGpuKernelMod::LaunchKernel<half>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat32)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat32),
    &SparseReorderGpuKernelMod::LaunchKernel<float>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeFloat64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeFloat64),
    &SparseReorderGpuKernelMod::LaunchKernel<double>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex64)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex64),
    &SparseReorderGpuKernelMod::LaunchKernel<cuFloatComplex>},
   {KernelAttr()
      .AddInputAttr(kNumberTypeInt64)
      .AddInputAttr(kNumberTypeComplex128)
      .AddInputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeInt64)
      .AddOutputAttr(kNumberTypeComplex128),
    &SparseReorderGpuKernelMod::LaunchKernel<cuDoubleComplex>}};

std::vector<KernelAttr> SparseReorderGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseReorderFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseReorder, SparseReorderGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
