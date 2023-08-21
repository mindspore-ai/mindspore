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

#include <functional>
#include "plugin/device/gpu/kernel/sparse/sparse_fill_empty_rows_gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
bool SparseFillEmptyRowsGpuKernelMod::Init(const BaseOperatorPtr &base_operator,
                                           const std::vector<KernelTensorPtr> &inputs,
                                           const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::SparseFillEmptyRows>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast SparseFillEmptyRows ops failed!";
    return false;
  }
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kSparseFillEmptyRowsInputsNum, kernel_ptr->name());
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kSparseFillEmptyRowsOutputsNum, kernel_ptr->name());

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_EXCEPTION(TypeError) << "For '" << kernel_ptr->name()
                            << "', it does not support this kernel data type: " << kernel_attr;
  }
  if (abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) != abstract::TypeIdSize(inputs[kIndex3]->GetDtype())) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_ptr->name()
                             << "The datatypes of values and default_value are not same.";
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

void SparseFillEmptyRowsGpuKernelMod::ResetResource() noexcept {
  is_null_input_ = false;
  input_dense_shape_size_ = 0;
  input_indice_size_ = 0;
  input_values_size_ = 0;
  input_default_values_size_ = 0;
  output_indices_size_ = 0;
  output_values_size_ = 0;
  output_reverse_index_map_size_ = 0;
  output_empty_row_indicator_size_ = 0;
  output_elements1_ = 0;
  output_elements2_ = 0;
  output_elements3_ = 0;
  output_elements4_ = 0;
  dense_row = 0;
  real_output_size_ = 0;
  input_size_list_.clear();
  workspace_size_list_.clear();
  output_size_list_.clear();
}

int SparseFillEmptyRowsGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                            const std::vector<KernelTensorPtr> &inputs,
                                            const std::vector<KernelTensorPtr> &outputs,
                                            const std::map<uint32_t, tensor::TensorPtr> &others) {
  is_need_retrieve_output_shape_ = true;  // infershape dynamic in gpu kernel.
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  if (!IsValidShape(outputs[kIndex0]->GetMaxShape()) || !IsValidShape(outputs[kIndex1]->GetMaxShape())) {
    return KRET_UNKNOWN_SHAPE;
  }
  if (!IsValidShape(outputs[kIndex2]->GetShapeVector()) || !IsValidShape(outputs[kIndex3]->GetShapeVector())) {
    return KRET_UNKNOWN_SHAPE;
  }
  ResetResource();
  input_indices_shapes_ = inputs[kIndex0]->GetShapeVector();
  input_values_shapes_ = inputs[kIndex1]->GetShapeVector();
  input_dense_shape_shapes_ = inputs[kIndex2]->GetShapeVector();
  output_indices_shapes_ = outputs[kIndex0]->GetMaxShape();
  output_values_shapes_ = outputs[kIndex1]->GetMaxShape();
  output_empty_row_indicator_shape_ = outputs[kIndex2]->GetShapeVector();
  output_reverse_index_map_shape_ = outputs[kIndex3]->GetShapeVector();
  if (inputs[kIndex0]->GetShapeVector()[0] != inputs[kIndex1]->GetShapeVector()[0]) {
    MS_EXCEPTION(ValueError) << "The element number of indices should be equal to values element number.";
  }
  input_default_values_size_ = abstract::TypeIdSize(inputs[kIndex3]->GetDtype());
  input_indice_size_ =
    abstract::TypeIdSize(inputs[kIndex0]->GetDtype()) * input_indices_shapes_[kIndex0] * input_indices_shapes_[kIndex1];
  input_values_size_ = abstract::TypeIdSize(inputs[kIndex1]->GetDtype()) * input_values_shapes_[kIndex0];
  input_dense_shape_size_ = abstract::TypeIdSize(inputs[kIndex2]->GetDtype()) * input_dense_shape_shapes_[kIndex0];
  output_elements1_ =
    std::accumulate(output_indices_shapes_.begin(), output_indices_shapes_.end(), 1, std::multiplies<int64_t>());
  output_elements2_ =
    std::accumulate(output_values_shapes_.begin(), output_values_shapes_.end(), 1, std::multiplies<int64_t>());
  output_elements3_ = std::accumulate(output_empty_row_indicator_shape_.begin(),
                                      output_empty_row_indicator_shape_.end(), 1, std::multiplies<int64_t>());
  output_elements4_ = std::accumulate(output_reverse_index_map_shape_.begin(), output_reverse_index_map_shape_.end(), 1,
                                      std::multiplies<int64_t>());
  dense_row = output_elements3_;
  if (output_elements1_ == 0 || output_elements2_ == 0 || output_elements3_ == 0 || output_elements4_ == 0) {
    is_null_input_ = true;
  }
  output_indices_size_ = abstract::TypeIdSize(outputs[kIndex0]->GetDtype()) * output_elements1_;
  output_values_size_ = abstract::TypeIdSize(outputs[kIndex1]->GetDtype()) * output_elements2_;
  auto output_empty_row_indicator_type = outputs[kIndex2]->GetDtype();
  output_empty_row_indicator_size_ = abstract::TypeIdSize(output_empty_row_indicator_type) * output_elements3_;
  auto output_reverse_index_map_type = outputs[kIndex3]->GetDtype();
  output_reverse_index_map_size_ = abstract::TypeIdSize(output_reverse_index_map_type) * output_elements4_;

  auto workspace_elements_per_rows_size = dense_row * sizeof(int64_t);
  auto workspace_empty_rows_count_size = dense_row * sizeof(int);
  auto workspace_row_indices_size = input_indice_size_ * sizeof(int64_t);
  auto workspace_input_row_ends_size = dense_row * sizeof(int64_t);
  auto workspace_sorted_indices_size = input_indice_size_ * sizeof(int64_t);
  auto workspace_final_shape_size = sizeof(size_t);
  auto workspace_origin_index_order_size = input_indices_shapes_[kIndex0] * sizeof(int64_t);
  auto workspace_sorted_key_size = input_indices_shapes_[kIndex0] * sizeof(int64_t);

  input_size_list_.push_back(input_indice_size_);
  input_size_list_.push_back(input_values_size_);
  input_size_list_.push_back(input_dense_shape_size_);
  input_size_list_.push_back(input_default_values_size_);

  workspace_size_list_.push_back(workspace_elements_per_rows_size);
  workspace_size_list_.push_back(workspace_empty_rows_count_size);
  workspace_size_list_.push_back(workspace_row_indices_size);
  workspace_size_list_.push_back(workspace_input_row_ends_size);
  workspace_size_list_.push_back(workspace_sorted_indices_size);
  workspace_size_list_.push_back(workspace_final_shape_size);
  workspace_size_list_.push_back(workspace_origin_index_order_size);
  workspace_size_list_.push_back(workspace_sorted_key_size);

  output_size_list_.push_back(output_indices_size_);
  output_size_list_.push_back(output_values_size_);
  output_size_list_.push_back(output_empty_row_indicator_size_);
  output_size_list_.push_back(output_reverse_index_map_size_);
  return KRET_OK;
}

template <typename S>
bool SparseFillEmptyRowsGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                                   const std::vector<AddressPtr> &workspace,
                                                   const std::vector<AddressPtr> &outputs) {
  if (is_null_input_) {
    return true;
  }

  auto input_indice_addr = GetDeviceAddress<int64_t>(inputs, kIndex0);
  auto input_values_addr = GetDeviceAddress<S>(inputs, kIndex1);
  auto input_dense_shape_addr = GetDeviceAddress<int64_t>(inputs, kIndex2);
  auto input_default_values_addr = GetDeviceAddress<S>(inputs, kIndex3);
  auto workspace_elements_per_rows_addr = GetDeviceAddress<int64_t>(workspace, kIndex0);
  auto workspace_empty_rows_count_addr = GetDeviceAddress<int>(workspace, kIndex1);
  auto workspace_row_indices_addr = GetDeviceAddress<int64_t>(workspace, kIndex2);
  auto workspace_input_row_ends_addr = GetDeviceAddress<int64_t>(workspace, kIndex3);
  auto workspace_sorted_indices_addr = GetDeviceAddress<int64_t>(workspace, kIndex4);
  auto workspace_final_shape_addr = GetDeviceAddress<size_t>(workspace, kIndex5);
  auto workspace_origin_index_addr = GetDeviceAddress<int64_t>(workspace, kIndex6);
  auto workspace_sorted_key_addr = GetDeviceAddress<int64_t>(workspace, kIndex7);
  auto output_indices_addr = GetDeviceAddress<int64_t>(outputs, kIndex0);
  auto output_values_addr = GetDeviceAddress<S>(outputs, kIndex1);
  auto output_empty_row_indicator_addr = GetDeviceAddress<bool>(outputs, kIndex2);
  auto output_reverse_index_map_addr = GetDeviceAddress<int64_t>(outputs, kIndex3);

  MS_EXCEPTION_IF_NULL(input_indice_addr);
  MS_EXCEPTION_IF_NULL(input_values_addr);
  MS_EXCEPTION_IF_NULL(input_dense_shape_addr);
  MS_EXCEPTION_IF_NULL(input_default_values_addr);
  MS_EXCEPTION_IF_NULL(workspace_elements_per_rows_addr);
  MS_EXCEPTION_IF_NULL(workspace_empty_rows_count_addr);
  MS_EXCEPTION_IF_NULL(workspace_input_row_ends_addr);
  MS_EXCEPTION_IF_NULL(workspace_row_indices_addr);
  MS_EXCEPTION_IF_NULL(workspace_sorted_indices_addr);
  MS_EXCEPTION_IF_NULL(workspace_final_shape_addr);
  MS_EXCEPTION_IF_NULL(workspace_origin_index_addr);
  MS_EXCEPTION_IF_NULL(workspace_sorted_key_addr);
  MS_EXCEPTION_IF_NULL(output_indices_addr);
  MS_EXCEPTION_IF_NULL(output_values_addr);
  MS_EXCEPTION_IF_NULL(output_empty_row_indicator_addr);
  MS_EXCEPTION_IF_NULL(output_reverse_index_map_addr);

  auto status =
    SparseFillEmptyRows(input_indice_addr, input_values_addr, input_default_values_addr, input_dense_shape_addr, 0,
                        input_indices_shapes_[0], dense_row, workspace_elements_per_rows_addr,
                        workspace_empty_rows_count_addr, workspace_row_indices_addr, workspace_input_row_ends_addr,
                        workspace_sorted_indices_addr, workspace_final_shape_addr, workspace_origin_index_addr,
                        workspace_sorted_key_addr, reinterpret_cast<cudaStream_t>(cuda_stream_), output_indices_addr,
                        output_values_addr, output_empty_row_indicator_addr, output_reverse_index_map_addr);
  CHECK_CUDA_STATUS(status, kernel_name_);
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_size_, workspace_final_shape_addr, sizeof(int64_t), cudaMemcpyDeviceToHost,
                    reinterpret_cast<cudaStream_t>(cuda_stream_)),
    "SparseFillEmptyRows cudaMemcpyAsync failed.");
  if (cudaStreamQuery(reinterpret_cast<cudaStream_t>(cuda_stream_)) != cudaSuccess) {
    CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(cuda_stream_)),
                                       "For 'SparseFillEmptyRows', cuda Stream Sync Failed.");
  }
  return true;
}

void SparseFillEmptyRowsGpuKernelMod::SyncOutputShape() {
  std::vector<int64_t> new_output_indice_shape = {SizeToLong(real_output_size_), 2};
  outputs_[kIndex0]->SetShapeVector(new_output_indice_shape);
  std::vector<int64_t> new_output_values_shape = {SizeToLong(real_output_size_)};
  outputs_[kIndex1]->SetShapeVector(new_output_values_shape);
  outputs_[kIndex2]->SetShapeVector(output_empty_row_indicator_shape_);
  outputs_[kIndex3]->SetShapeVector(output_reverse_index_map_shape_);
}
std::vector<KernelAttr> SparseFillEmptyRowsGpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseFillEmptyRowsFunc> &pair) { return pair.first; });
  return support_list;
}

template <typename S>
using Complex = mindspore::utils::Complex<S>;
std::vector<std::pair<KernelAttr, SparseFillEmptyRowsGpuKernelMod::SparseFillEmptyRowsFunc>>
  SparseFillEmptyRowsGpuKernelMod::func_list_ = {
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<bool>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<uint16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<uint8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<uint32_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<uint64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<int8_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<int16_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<int>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<half>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<float>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<double>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<Complex<float>>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseFillEmptyRowsGpuKernelMod::LaunchKernel<Complex<double>>},
};

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseFillEmptyRows, SparseFillEmptyRowsGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
