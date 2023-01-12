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

#include "mindspore/ccsrc/plugin/device/gpu/kernel/sparse/sparse_slice_gpu_kernel.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/complex.h"

namespace mindspore {
namespace kernel {
template <typename T>
using Complex = mindspore::utils::Complex<T>;
bool SparseSliceGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                   const std::vector<KernelTensorPtr> &outputs) {
  constexpr size_t inputs_num = 5;
  constexpr size_t outputs_num = 3;
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
  is_need_retrieve_output_shape_ = true;
  return true;
}

int SparseSliceGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                    const std::vector<KernelTensorPtr> &outputs,
                                    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  auto ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost);
  if (ret == KRET_UNKNOWN_OUT_SHAPE) {
    outputs_ = outputs;
    auto input_indices_shape = inputs[kIndex0]->GetShapeVector();
    auto out_shape = outputs.at(kIndex2)->GetShapeVector();
    auto out_size = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int64_t>());
    out_size_ = out_size;

    input_nnz_ = input_indices_shape[0];
    num_dim_ = input_indices_shape[1];

    output_size_list_.clear();
    output_size_list_.emplace_back(input_nnz_ * num_dim_ * GetTypeByte(TypeIdToType(inputs[kIndex0]->GetDtype())));
    output_size_list_.emplace_back(input_nnz_ * GetTypeByte(TypeIdToType(inputs[kIndex1]->GetDtype())));
    output_size_list_.emplace_back(num_dim_ * GetTypeByte(TypeIdToType(inputs[kIndex2]->GetDtype())));

    workspace_size_list_.clear();
    workspace_size_list_.push_back(sizeof(int64_t));
  }
  return ret;
}

template <typename DataType, typename IndexType>
bool SparseSliceGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                           const std::vector<AddressPtr> &workspace,
                                           const std::vector<AddressPtr> &outputs, void *stream_ptr) {
  auto cuda_stream = reinterpret_cast<cudaStream_t>(stream_ptr);
  MS_EXCEPTION_IF_NULL(cuda_stream);
  auto indices_ptr = GetDeviceAddress<IndexType>(inputs, kIndex0);
  auto values_ptr = GetDeviceAddress<DataType>(inputs, kIndex1);
  auto x_ptr = GetDeviceAddress<IndexType>(inputs, kIndex2);
  auto start_ptr = GetDeviceAddress<IndexType>(inputs, kIndex3);
  auto size_ptr = GetDeviceAddress<IndexType>(inputs, kIndex4);
  auto y_indices_ptr = GetDeviceAddress<IndexType>(outputs, kIndex0);
  auto y_values_ptr = GetDeviceAddress<DataType>(outputs, kIndex1);
  auto out_shape_ptr = GetDeviceAddress<IndexType>(outputs, kIndex2);
  auto sum_count_ptr = GetDeviceAddress<int64_t>(workspace, kIndex0);

  CHECK_CUDA_RET_WITH_ERROR_NOTRACE(
    cudaMemsetAsync(sum_count_ptr, static_cast<int64_t>(0), workspace.at(kIndex0)->size, cuda_stream),
    "For SparseSlice, failed to cudaMemset.");

  bool is_nullptr = (indices_ptr == nullptr) || (values_ptr == nullptr) || (x_ptr == nullptr) ||
                    (start_ptr == nullptr) || (size_ptr == nullptr) || (y_indices_ptr == nullptr) ||
                    (y_values_ptr == nullptr) || (out_shape_ptr == nullptr) || (sum_count_ptr == nullptr);
  if (is_nullptr) {
    return false;
  }

  SparseSlice(indices_ptr, values_ptr, x_ptr, start_ptr, size_ptr, y_indices_ptr, y_values_ptr, out_shape_ptr,
              sum_count_ptr, input_nnz_, num_dim_, out_size_, device_id_, cuda_stream);

  // Get dynamic shape
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(&real_output_size, sum_count_ptr, sizeof(int64_t), cudaMemcpyDeviceToHost, cuda_stream),
    "For SparseSlice, cudaMemcpyAsync failed.");
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(
    cudaMemcpyAsync(out_shape_ptr, size_ptr, sizeof(int64_t) * out_size_, cudaMemcpyDeviceToDevice, cuda_stream),
    "For SparseSlice, cudaMemcpyAsync out_shape failed.");
  return true;
}

void SparseSliceGpuKernelMod::SyncData() {
  CHECK_CUDA_RET_WITH_EXCEPT_NOTRACE(cudaStreamSynchronize(cuda_stream), "SparseSlice cudaStreamSynchronized failed");
  outputs_[kIndex0]->SetShapeVector(ShapeVector({real_output_size, static_cast<int64_t>(num_dim_)}));
  outputs_[kIndex1]->SetShapeVector(ShapeVector({real_output_size}));
  outputs_[kIndex2]->SetShapeVector(ShapeVector({static_cast<int64_t>(num_dim_)}));
}

std::vector<std::pair<KernelAttr, SparseSliceGpuKernelMod::SparseSliceLaunchFunc>> SparseSliceGpuKernelMod::func_list_ =
  {{
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt8)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<uint8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt16)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<uint16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<uint32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeUInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeUInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<uint64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<int64_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<int32_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt16)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<int16_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt8)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt8)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<int8_t, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<double, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat32)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<float, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeFloat16)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat16)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<half, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeBool)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeBool)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<bool, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex64)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<Complex<float>, int64_t>},
    {KernelAttr()
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeComplex128)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddInputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeComplex128)
       .AddOutputAttr(kNumberTypeInt64),
     &SparseSliceGpuKernelMod::LaunchKernel<Complex<double>, int64_t>},
  }};

std::vector<KernelAttr> SparseSliceGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(
    func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
    [](const std::pair<KernelAttr, SparseSliceGpuKernelMod::SparseSliceLaunchFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, SparseSlice, SparseSliceGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
