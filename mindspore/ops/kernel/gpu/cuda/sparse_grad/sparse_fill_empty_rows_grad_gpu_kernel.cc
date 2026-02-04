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

#include "kernel/gpu/cuda/sparse_grad/sparse_fill_empty_rows_grad_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool SparseFillEmptyRowsGradGpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kInputsNum || outputs.size() != kOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', input and output must be " << kInputsNum << " and " << kOutputsNum
                  << ", but got " << inputs.size() << " and " << outputs.size();
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseFillEmptyRowsGradGpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  reverse_map_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeVector().begin(),
                                            inputs.at(kIndex0)->GetDeviceShapeVector().end());
  int64_t reverse_map_dims = reverse_map_shape_.size();
  if (reverse_map_dims != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'reverse_index_map' must be 1-D, but got "
                  << reverse_map_dims << "-D.";
    return false;
  }
  std::vector<int64_t> grad_values_shape = std::vector<int64_t>(inputs.at(kIndex1)->GetDeviceShapeVector().begin(),
                                                                inputs.at(kIndex1)->GetDeviceShapeVector().end());
  int64_t grad_values_dims = grad_values_shape.size();
  if (grad_values_dims != 1) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the dimension of 'grad_values' must be 1-D, but got "
                  << grad_values_dims << "-D.";
    return false;
  }
  reverse_map_shape_ = inputs[kIndex0]->GetShapeVector();
  reverse_map_num_ = reverse_map_shape_[0];
  grad_values_shapes_ = inputs[kIndex1]->GetShapeVector();
  grad_values_num_ = grad_values_shapes_[0];
  dvalues_shapes_ = outputs[kIndex0]->GetShapeVector();
  dvalues_num_ = dvalues_shapes_[0];
  if (dvalues_num_ == 0) {
    is_null_input_ = true;
  }

  output_dvalues_size_ = reverse_map_num_ * abstract::TypeIdSize(outputs[kIndex0]->dtype_id());
  output_ddefault_value_size_ = abstract::TypeIdSize(outputs[kIndex1]->dtype_id());
  reverse_map_size_ = reverse_map_num_ * abstract::TypeIdSize(inputs[kIndex0]->dtype_id());
  grad_values_size_ = grad_values_num_ * abstract::TypeIdSize(inputs[kIndex1]->dtype_id());
  workspace_flag_size_ = grad_values_num_ * sizeof(bool);
  workspace_sum_val_size_ =
    grad_values_num_ * abstract::TypeIdSize(inputs[kIndex1]->dtype_id());  // Precision need auxlilary memory

  workspace_size_list_.push_back(workspace_flag_size_);
  workspace_size_list_.push_back(workspace_sum_val_size_);
  output_size_list_.push_back(output_dvalues_size_);
  output_size_list_.push_back(output_ddefault_value_size_);
  return KRET_OK;
}

template <typename T>
bool SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &workspace,
                                                       const std::vector<KernelTensor *> &outputs, void *cuda_stream) {
  int64_t *reverse_index_map = GetDeviceAddress<int64_t>(inputs, 0);
  T *grad_values = GetDeviceAddress<T>(inputs, 1);
  bool *workspace_flag = GetDeviceAddress<bool>(workspace, 0);
  void *workspace_sum_val = GetDeviceAddress<void>(workspace, 1);
  T *d_values = GetDeviceAddress<T>(outputs, 0);
  T *d_default_value = GetDeviceAddress<T>(outputs, 1);

  auto status =
    CalFillRowsGrad(reverse_map_num_, grad_values_num_, reverse_index_map, grad_values, d_values, d_default_value,
                    workspace_flag, workspace_sum_val, device_id_, reinterpret_cast<cudaStream_t>(cuda_stream));
  CHECK_CUDA_STATUS(status, kernel_name_);
  return true;
}

template <typename T>
using Complex = mindspore::utils::Complex<T>;
std::vector<std::pair<KernelAttr, SparseFillEmptyRowsGradGpuKernelMod::SparseFillEmptyRowsGradFunc>>
  SparseFillEmptyRowsGradGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeUInt32),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseFillEmptyRowsGradGpuKernelMod::LaunchKernel<Complex<double>>}};

std::vector<KernelAttr> SparseFillEmptyRowsGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseFillEmptyRowsGradFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseFillEmptyRowsGrad, SparseFillEmptyRowsGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
