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

#include "mindspore/ccsrc/plugin/device/gpu/kernel/sparse_grad/sparse_slice_grad_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool SparseSliceGradGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t inputs_num = 4;
  constexpr size_t outputs_num = 1;
  kernel_name_ = base_operator->GetPrim()->name();
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), inputs_num, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), outputs_num, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int SparseSliceGradGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = KernelMod::Resize(base_operator, inputs, outputs);
  if (ret != KRET_OK) {
    return ret;
  }

  auto x_shape = inputs.at(kIndex0)->GetShapeVector();
  auto x_size = std::accumulate(x_shape.begin(), x_shape.end(), 1, std::multiplies<size_t>());
  num_grad_val_ = x_size;
  size_t unit_size_ = abstract::TypeIdSize(inputs.at(kIndex0)->GetDtype());
  input_size_list_.push_back(x_size * unit_size_);

  auto indices_shape = inputs.at(kIndex1)->GetShapeVector();
  auto indices_size = std::accumulate(indices_shape.begin(), indices_shape.end(), 1, std::multiplies<size_t>());
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex1)->GetDtype());
  input_size_list_.push_back(indices_size * unit_size_);
  input_nnz_ = indices_shape[0];
  num_dim_ = indices_shape[1];

  auto start_shape = inputs.at(kIndex2)->GetShapeVector();
  auto start_size = std::accumulate(start_shape.begin(), start_shape.end(), 1, std::multiplies<size_t>());
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex2)->GetDtype());
  input_size_list_.push_back(start_size * unit_size_);

  auto new_indices_shape = inputs.at(kIndex3)->GetShapeVector();
  auto new_indices_size =
    std::accumulate(new_indices_shape.begin(), new_indices_shape.end(), 1, std::multiplies<size_t>());
  unit_size_ = abstract::TypeIdSize(inputs.at(kIndex3)->GetDtype());
  input_size_list_.push_back(new_indices_size * unit_size_);
  output_nnz_ = new_indices_shape[0];

  auto input_indices_shape = inputs.at(kIndex1)->GetShapeVector();
  const int64_t input_nnz = input_indices_shape[0];
  unit_size_ = abstract::TypeIdSize(outputs.at(kIndex0)->GetDtype());
  output_size_list_.clear();
  output_size_list_.push_back(input_nnz * unit_size_);

  workspace_size_list_.clear();
  workspace_size_list_.push_back(sizeof(size_t));
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSliceGradGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  auto x_ptr = GetDeviceAddress<DataType>(inputs, kIndex0);
  auto indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex1);
  auto start_ptr = GetDeviceAddress<IndexType>(inputs, kIndex2);
  auto new_indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex3);
  auto y_ptr = GetDeviceAddress<DataType>(outputs, kIndex0);
  auto num_propagated_ptr = GetDeviceAddress<size_t>(workspace, kIndex0);
  bool is_nullptr = (x_ptr == nullptr) || (indices_ptr == nullptr) || (start_ptr == nullptr) ||
                    (new_indices_ptr == nullptr) || (y_ptr == nullptr) || (num_propagated_ptr == nullptr);
  if (is_nullptr) {
    return false;
  }

  size_t num_propagated = 0;
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(num_propagated_ptr, &num_propagated, sizeof(size_t), cudaMemcpyHostToDevice, cuda_stream),
    "cudaMemcpyHostToDevice for 'SparseSliceGrad' num_propagated failed");

  SparseSliceGrad(x_ptr, indices_ptr, start_ptr, new_indices_ptr, y_ptr, num_propagated_ptr, input_nnz_, output_nnz_,
                  num_dim_, device_id_, cuda_stream);

  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&num_propagated, num_propagated_ptr, sizeof(size_t), cudaMemcpyDeviceToHost, cuda_stream),
    "cudaMemcpyDeviceToHost for 'SparseSliceGrad' num_propagated failed");

  if (num_propagated == num_grad_val_) {
    return true;
  } else {
    MS_LOG(ERROR) << kernel_name_ << " Elements of backprop_val_grad are not all propagated. "
                  << "Num elements:" << num_grad_val_ << ", used: " << num_propagated;
    return false;
  }
}

std::vector<std::pair<KernelAttr, SparseSliceGradGpuKernelMod::SparseSliceGradLaunchFunc>>
  SparseSliceGradGpuKernelMod::func_list_ = {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8),
     &SparseSliceGradGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16),
     &SparseSliceGradGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &SparseSliceGradGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGradGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8),
     &SparseSliceGradGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16),
     &SparseSliceGradGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32),
     &SparseSliceGradGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64),
     &SparseSliceGradGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16),
     &SparseSliceGradGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &SparseSliceGradGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64),
     &SparseSliceGradGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool),
     &SparseSliceGradGpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64),
     &SparseSliceGradGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128),
     &SparseSliceGradGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
  }};

std::vector<KernelAttr> SparseSliceGradGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, SparseSliceGradGpuKernelMod::SparseSliceGradLaunchFunc> &pair) {
                         return pair.first;
                       });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSliceGrad, SparseSliceGradGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
